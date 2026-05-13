# Third Party Notices

This project includes third-party software packages. Their licenses remain the property of their respective authors.

## Runtime and UI (NuGet)

- Avalonia (`Avalonia`, `Avalonia.Desktop`, `Avalonia.Controls.DataGrid`, `Avalonia.Themes.Fluent`)
- SkiaSharp and native assets (transitive through Avalonia)
- HarfBuzzSharp (transitive through Avalonia/Skia)

## Python backend dependencies

- NumPy (`numpy`)
- PyTorch (`torch`) installed by the app setup based on detected hardware

## Distribution notes

- Do not distribute datasets, checkpoints, or model weights unless you verified their license permits redistribution.
- Keep this file in source and in binary releases.

## How to regenerate/update dependency license inventory

From repository root:

```bash
dotnet tool install --global dotnet-project-licenses
dotnet-project-licenses -i src/LLMForgeStudio.App/LLMForgeStudio.App.csproj
```

From `backends/python`:

```bash
pip install pip-licenses
pip-licenses
```
