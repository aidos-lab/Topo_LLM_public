import compute_euclidicity
import local_pca
import hydra

@hydra.main(
    config_path="../../configs/analysis",
    config_name="comparison",
    version_base="1.2",
)
def main(cfg):
    local_pca.main(cfg)
    compute_euclidicity.main(cfg)

    return None

if __name__ == "__main__":
    main()  # type: ignore