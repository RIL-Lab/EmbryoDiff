from data.MFHE.video_dataset import EmbryoVideoMFHE, VideoTransformMFHE
from data.SFHE.video_dataset import EmbryoVideoSFHE, VideoTransformSFHE
from data.MFHE.image_dataset import EmbryoImageMFHE, ImageTransformMFHE
from data.SFHE.image_dataset import EmbryoImageSFHE, ImageTransformSFHE


def build_train_test_set(cfg):

    dataset_name = cfg["data"].get("dataset_name", None)
    if dataset_name is None:
        raise ValueError("dataset_name is None.")
    
    elif dataset_name == "MFHE":
        train_set = EmbryoVideoMFHE(split_file=cfg["data"]["train_split"], 
                                                anno_root=cfg["data"]["anno_root"], 
                                                video_feat_folder=cfg["data"]["video_feat_folder"], 
                                                transform=VideoTransformMFHE(),
                                                subsample_rate=cfg["train"]["subsample_rate"],
                                                num_classes=cfg["data"]["num_classes"],
                                                fast_load=cfg["train"]["debug"],
                                                boundary_smooth=cfg["train"]["boundary_smooth"]
                                                )
        
        test_set = EmbryoVideoMFHE(split_file=cfg["data"]["test_split"], 
                                                anno_root=cfg["data"]["anno_root"], 
                                                video_feat_folder=cfg["data"]["video_feat_folder"],
                                                transform=VideoTransformMFHE(),
                                                subsample_rate=cfg["train"]["subsample_rate"],
                                                num_classes=cfg["data"]["num_classes"],
                                                fast_load=cfg["train"]["debug"],
                                                boundary_smooth=cfg["train"]["boundary_smooth"]
                                                )

    
    elif dataset_name == "SFHE":
        train_set = EmbryoVideoSFHE(
                                    anno_path=cfg["data"]["train_split"], 
                                    video_feat_folder=cfg["data"]["video_feat_folder"], 
                                    development_stage_list=cfg["data"]["development_stage_list"],
                                    transform=VideoTransformSFHE(),
                                    num_classes=cfg["data"]["num_classes"],
                                    fast_load=cfg["train"]["debug"]
                                    )
        test_set = EmbryoVideoSFHE(
                                    anno_path=cfg["data"]["test_split"], 
                                    video_feat_folder=cfg["data"]["video_feat_folder"],
                                    development_stage_list=cfg["data"]["development_stage_list"],
                                    transform=VideoTransformSFHE(),
                                    num_classes=cfg["data"]["num_classes"],
                                    fast_load=cfg["train"]["debug"]
                                    )

    else:
        raise ValueError(f"dataset {dataset_name} if not supported.")
    

    return train_set, test_set


def build_train_test_set_stage1(config):

    dataset_name = config.get("dataset_name", None)
    if dataset_name is None:
        raise ValueError("dataset_name is None.")
    
    if dataset_name == "MFHE":
        train_transform = ImageTransformMFHE(mode="train")
        train_set = EmbryoImageMFHE(
            split_file=config["train_split"],
            anno_root=config["anno_root"],
            video_root=config["video_root"],
            focal_planes=config["focal_planes"],
            num_classes=config["num_classes"],
            subsample_rate=config["subsample_rate"],
            transform=train_transform,
            debug=config["debug"]
        )

        test_transform = ImageTransformMFHE(mode="test")
        test_set = EmbryoImageMFHE(
            split_file=config["test_split"],
            anno_root=config["anno_root"],
            video_root=config["video_root"],
            focal_planes=config["focal_planes"],
            num_classes=config["num_classes"],
            subsample_rate=config["subsample_rate"],
            transform=test_transform,
            debug=config["debug"]
        )
    
    elif dataset_name == "SFHE":
        train_transform = ImageTransformSFHE(mode="train")
        train_set = EmbryoImageSFHE(
            anno_path=config["train_split"],
            video_root=config["video_root"],
            num_classes=config["num_classes"],
            transform=train_transform
        )

        test_transform = ImageTransformSFHE(mode="test")
        test_set = EmbryoImageSFHE(
            anno_path=config["test_split"],
            video_root=config["video_root"],
            num_classes=config["num_classes"],
            transform=test_transform
        )

    else:
        raise ValueError(f"dataset {dataset_name} if not supported.")

    return train_set, test_set