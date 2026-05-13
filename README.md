# LLM Forge Studio

Desktop app (**.NET 8 + Avalonia**) to train and test local mini-LLMs on Windows with a guided click-first workflow.

## Platform Support

- Primary target: **Windows**
- Linux/macOS: **experimental** (manual backend setup may be required, automatic GPU setup is currently Windows-focused)

## Why LLM Forge Studio

- End-to-end local workflow: dataset -> tokenization -> model -> training -> generation
- GPU/CPU backend setup from UI
- Live training stats, overfitting signal, and quality score
- Project save/load with reproducible run settings
- One-click Windows release artifact via GitHub Actions

## Architecture

- `src/LLMForgeStudio.App`: Avalonia UI + C# core (dataset, tokenization, training utilities, sampling, project persistence)
- `backends/python`: PyTorch backend for training and generation
- `tests/LLMForgeStudio.App.Tests`: core unit tests

## Run From Source

```bash
dotnet restore
dotnet run --project src/LLMForgeStudio.App
```

## Python Backend Setup

```bash
cd backends/python
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

In the Training section of the UI:
- set `Python` to your interpreter path (for example `.venv\Scripts\python.exe`)
- set `Run directory`
- click `Start Backend Training`

## Use The Windows EXE

Download the latest Windows build from GitHub Releases and run:

```text
LLMForgeStudio.App.exe
```

## Build Windows EXE DIY (manual)

```bash
dotnet publish src/LLMForgeStudio.App/LLMForgeStudio.App.csproj -c Release -r win-x64 --self-contained true /p:PublishSingleFile=true /p:IncludeNativeLibrariesForSelfExtract=true
```

Output:

```text
src/LLMForgeStudio.App/bin/Release/net8.0/win-x64/publish/LLMForgeStudio.App.exe
```
