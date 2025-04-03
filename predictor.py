from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from PIL import Image
import numpy as np
import os

class LeffaPredictor:
    def __init__(self):
        ckpts_root = os.path.join(os.path.dirname(__file__), "ckpts")

        self.mask_predictor = AutoMasker(
            densepose_path=os.path.join(ckpts_root, "densepose"),
            schp_path=os.path.join(ckpts_root, "schp"),
        )

        self.densepose_predictor = DensePosePredictor(
            config_path=os.path.join(ckpts_root, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
            weights_path=os.path.join(ckpts_root, "densepose", "model_final_162be9.pkl"),
        )

        self.parsing = Parsing(
            atr_path=os.path.join(ckpts_root, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(ckpts_root, "humanparsing", "parsing_lip.onnx"),
        )

        self.openpose = OpenPose(
            body_model_path=os.path.join(ckpts_root, "openpose", "body_pose_model.pth"),
        )

        self.vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=os.path.join(ckpts_root, "stable-diffusion-inpainting"),
            pretrained_model=os.path.join(ckpts_root, "virtual_tryon.pth"),
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=self.vt_model_hd)

        self.vt_model_dc = LeffaModel(
            pretrained_model_name_or_path=os.path.join(ckpts_root, "stable-diffusion-inpainting"),
            pretrained_model=os.path.join(ckpts_root, "virtual_tryon_dc.pth"),
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=self.vt_model_dc)

        self.pt_model = LeffaModel(
            pretrained_model_name_or_path=os.path.join(ckpts_root, "stable-diffusion-xl-1.0-inpainting-0.1"),
            pretrained_model=os.path.join(ckpts_root, "pose_transfer.pth"),
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=self.pt_model)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration=False, step=50, scale=2.5, seed=42, vt_model_type="viton_hd", vt_garment_type="upper_body", vt_repaint=False, preprocess_garment=False):
        return self._predict(src_image_path, ref_image_path, "virtual_tryon", ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment)

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration=False, step=50, scale=2.5, seed=42):
        return self._predict(src_image_path, ref_image_path, "pose_transfer", ref_acceleration, step, scale, seed)

    def _predict(self, src_image_path, ref_image_path, control_type, ref_acceleration=False, step=50, scale=2.5, seed=42, vt_model_type="viton_hd", vt_garment_type="upper_body", vt_repaint=False, preprocess_garment=False):
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)

        src_image = resize_and_center(src_image, 768, 1024)

        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith(".png"):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            else:
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        else:
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(seg_array)
            else:
                iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                seg_array = np.concatenate([iuv_array[:, :, 0:1]] * 3, axis=-1)
                densepose = Image.fromarray(seg_array)
        else:
            iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(iuv_array)

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        if control_type == "virtual_tryon":
            inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        else:
            inference = self.pt_inference

        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        generated_image = output["generated_image"][0]
        if isinstance(generated_image, Image.Image):
            generated_image = np.array(generated_image)
        return generated_image, mask, densepose
