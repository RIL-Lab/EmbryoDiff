from models.model import EmbryoDiff


def build_model(cfg, **kwargs):

    device = kwargs.get("device")

    model = EmbryoDiff(
        encoder_params=cfg["encoder_params"],
        decoder_params=cfg["decoder_params"],
        diffusion_params=cfg["diffusion_params"],
        num_classes=cfg["data"]["num_classes"],
        device=device
    )

    return model