import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
import tqdm
import os
import argparse

from typing import Union, List, Tuple

# Ref : https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94


class PanoramaProjector:
    """
    Projecting the panorama on a specific-view perspective image.
    Optimized by caching the camera grid and image tensor to speed up batch processing.
    """

    def __init__(
        self,
        panorama_input: Union[str, Image.Image, np.ndarray],
        output_size: Tuple[int, int] = (1024, 1024),
        fov: float = 120.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.W, self.H = output_size
        self.fov = fov

        # Load panorama image
        self.img_tensor = self._load_image_to_tensor(panorama_input)

        # Init grid: Calculate only once and cache it
        self.x, self.y, self.z = self._init_camera_grid()

    def _load_image_to_tensor(self, img_input) -> torch.Tensor:
        """
        Handle input types and normalize the panorama image.
        """
        if isinstance(img_input, str):
            # if given image path
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Error: File not found at {img_input}")
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            # if given image tensor
            img = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            # if given Image object
            img = img_input.convert("RGB")
        else:
            raise TypeError(
                "Unsupported image format. Use path string, PIL Image, or Numpy array."
            )

        # Normalize the panorama image
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        # Shape: [H,W,C] -> [1,C,H,W] for the purpose of sampling
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def _init_camera_grid(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given output image size W x H and fov degree, generate cartesian coordinate matrix for each image pixel.
        """
        u = torch.arange(self.W, device=self.device, dtype=torch.float32)
        v = torch.arange(self.H, device=self.device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")

        x = u_grid - self.W * 0.5
        y = self.H * 0.5 - v_grid

        fov_rad = torch.tensor(self.fov, device=self.device).deg2rad()
        f = (self.W * 0.5) / torch.tan(fov_rad * 0.5)
        z = torch.full_like(x, f)

        return x, y, z

    def _cartesian_to_spherical(self, yaw_rad: torch.Tensor, pitch_rad: torch.Tensor):
        """
        Function : Cartesian2Spherical
        Translate the camera coordinate to world coordinate by multiplying the rotation matrices
        Input:
            1. yaw_rad, pitch_rad : the radians of yaw angle and pitch angle
        Output:
            The spherical coordinate of each image pixel
        """
        # Shape of x,y,z : [H,W] which are the output size of projecting image
        # Shape: [3,N] , N = H*W , the number of pixel
        points = torch.stack(
            [self.x.flatten(), self.y.flatten(), self.z.flatten()], dim=0
        )

        # Rotation matrix
        # Pitch (X-axis)
        cosx, sinx = torch.cos(pitch_rad), torch.sin(pitch_rad)
        Rx = torch.tensor(
            [[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]], device=self.device
        )

        # Yaw (Y-axis)
        cosy, siny = torch.cos(yaw_rad), torch.sin(yaw_rad)
        Ry = torch.tensor(
            [[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]], device=self.device
        )

        # Combined Rotation matrix
        R = Ry @ Rx
        points_after_rotation = torch.matmul(R, points)
        x_prime, y_prime, z_prime = (
            points_after_rotation[0],
            points_after_rotation[1],
            points_after_rotation[2],
        )

        # Cartesian -> Spherical coordinate system
        r = torch.sqrt(x_prime**2 + y_prime**2 + z_prime**2)  # Radius
        phi = torch.atan2(x_prime, z_prime)
        theta = torch.asin(y_prime / (r + 1e-8))

        return theta.view(self.H, self.W), phi.view(self.H, self.W)

    def get_perspective(self, yaw_deg: float, pitch_deg: float) -> torch.Tensor:
        """
        Core projection logic for a single view.
        Returns: Tensor of shape [1, C, H, W]
        """
        # Degree -> radians
        yaw_radian = np.radians(yaw_deg)
        pitch_radian = np.radians(-pitch_deg)

        yaw_tensor = torch.tensor(yaw_radian, device=self.device, dtype=torch.float32)
        pitch_tensor = torch.tensor(
            pitch_radian, device=self.device, dtype=torch.float32
        )

        # Obtain the spherical coordinate of each projecting plane pixels
        theta, phi = self._cartesian_to_spherical(yaw_tensor, pitch_tensor)

        # Normalize
        u_grid = phi / torch.pi  # [-pi , pi] -> [-1,1]
        v_grid = -theta / (
            torch.pi / 2
        )  # [-pi/2 , pi/2] -> [-1,1] , negative here for translate 3D-plane into input panorama

        # (u,v) : As a ratio where the location of panorama we are going to sample
        # Ex: (0.5,0.5) , we should sample the right half and bottom half pixel of panorama
        # shape: [1,H,W,2] , spherical coordinate of each 2D-plane pixel
        grid = torch.stack((u_grid, v_grid), dim=-1).unsqueeze(0)

        # Sampling
        # mode could be bicubic or bilinear
        output_tensor = F.grid_sample(
            self.img_tensor,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        return output_tensor

    def get_perspectives_batch(
        self, yaws: List[float], pitches: List[float]
    ) -> torch.Tensor:
        """
        Returns: Tensor of shape [N, C, H, W] if multiple angles
        """
        # Handle input types: ensure they are iterable lists and match in length
        if len(yaws) != len(pitches):
            raise ValueError(
                f"Yaw and Pitch length mismatch: {len(yaws)} vs {len(pitches)}"
            )

        # Loop through angles
        outputs = [self.get_perspective(y, p) for y, p in zip(yaws, pitches)]

        # Stack results
        return torch.cat(outputs, dim=0)

    def export_video(
        self,
        output_video: str,
        fps: int = 60,
        batch_size: int = 10,
        target_yaws: np.ndarray = None,
    ):
        """
        Rendering Video
        """
        # Output yaw angle range from 0 ~ 359
        if target_yaws is None:
            # np.arange(start, stop, step)
            target_yaws = np.arange(0, 360, 1.0)

        fixed_pitch = 0.0
        print(f"Rendering Video, Total {len(target_yaws)} frames...")
       
        # Checking the output path. 
        output_dir = os.path.dirname(output_video)
        if output_dir:  # if path contains directories
            os.makedirs(output_dir, exist_ok=True)
            
        writer = imageio.get_writer(output_video, fps=fps, macro_block_size=None)

        # batch_size: 10 frames per iteration in case of out of VRAM
        for i in tqdm.tqdm(range(0, len(target_yaws), batch_size)):
            batch_yaws = target_yaws[i : i + batch_size]
            # Broadcast pitch to match length of yaw
            batch_pitches = [fixed_pitch] * len(batch_yaws)

            with torch.no_grad():
                batch_output = self.get_perspectives_batch(batch_yaws, batch_pitches)

            # Post-Processing (Tensor -> Numpy -> uint8)
            # [N, C, H, W] -> [N, H, W, C]
            batch_np = batch_output.permute(0, 2, 3, 1).cpu().numpy()
            # Constraint the data range 0~255 and turn them into integer
            batch_np = np.clip(batch_np * 255, 0, 255).astype(np.uint8)

            for frame in batch_np:
                writer.append_data(frame)

        writer.close()
        print(f"Saved Video: {output_video}")


if __name__ == "__main__":
    """
    Argument Parser :
    data_dir : location where the dataset is
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Location where the data is",
        type=str,
        default="./dollhouse_file/DollhouseTask_58472_NoOutdoor/DollhouseTask_58472_NoOutdoor/08ddb587-468c-45b7-8681-e213539a1710.jpg",
    )
    parser.add_argument(
        "--output_video_path",
        help="Output video path",
        type=str,
        default="./Video/output_360_rotation.mp4",
    )
    parser.add_argument("--fov", help="Field of view", default=120, type=float)
    parser.add_argument(
        "--output_size",
        help="Projecting image size",
        default=[1024, 1024],
        nargs="+",
        type=int,
    )
    parser.add_argument("--device", help="CPU or CUDA", default="cuda")
    args = parser.parse_args()

    print(f"Processing on {args.device}")

    # The size of projecting image
    W = args.output_size[0]
    H = args.output_size[0] if len(args.output_size) == 1 else args.output_size[1]

    # Initialize projector (Handles loading and grid initialization automatically)
    projector = PanoramaProjector(
        panorama_input=args.data_dir,
        output_size=(W, H),
        fov=args.fov,
        device=args.device,
    )

    # Render Video
    projector.export_video(args.output_video_path)
