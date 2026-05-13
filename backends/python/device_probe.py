import json


def main():
    try:
        import torch
    except Exception as e:
        print(json.dumps({"device": "none", "detail": f"torch import failed: {e}"}))
        return

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        print(json.dumps({"device": "cuda", "detail": name}))
        return

    try:
        import torch_directml
        dml = torch_directml.device()
        print(json.dumps({"device": "directml", "detail": str(dml)}))
        return
    except Exception:
        pass

    print(json.dumps({"device": "cpu", "detail": "fallback"}))


if __name__ == "__main__":
    main()
