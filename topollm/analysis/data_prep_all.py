from topollm.analysis import data_prep, data_prep_cls, data_prep_mean
import hydra
import omegaconf

@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    data_prep.main(config)
    data_prep_cls.main(config)
    data_prep_mean.main(config)

    return None

if __name__ == "__main__":
    main()  # type: ignore