def main() -> None:
    try:
        from src.pipeline import PhaseOnePipeline
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise ModuleNotFoundError(
                "OpenCV is missing from the active Python environment. Activate the project "
                "venv at .venv or run setup.ps1 before starting the app."
            ) from exc
        raise

    pipeline = PhaseOnePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
