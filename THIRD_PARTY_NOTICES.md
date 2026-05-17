# THIRD_PARTY_NOTICES

This project is distributed under the MIT License.
The following third-party components are used in source/build/runtime workflows.

## .NET Application Dependencies

### Avalonia
- Package: `Avalonia` (`12.0.0`)
- Package: `Avalonia.Controls.DataGrid` (`12.0.0`)
- Package: `Avalonia.Desktop` (`12.0.0`)
- Package: `Avalonia.Themes.Fluent` (`12.0.0`)
- License: MIT
- Project: https://github.com/AvaloniaUI/Avalonia

## .NET Test Dependencies

### Microsoft.NET.Test.Sdk
- Package: `Microsoft.NET.Test.Sdk` (`17.11.1`)
- License: Microsoft package license (open distribution for test tooling)
- Project: https://www.nuget.org/packages/Microsoft.NET.Test.Sdk

### xUnit
- Package: `xunit` (`2.9.2`)
- Package: `xunit.runner.visualstudio` (`2.8.2`)
- License: Apache-2.0
- Project: https://xunit.net/

### Coverlet
- Package: `coverlet.collector` (`6.0.2`)
- License: MIT
- Project: https://github.com/coverlet-coverage/coverlet

## Python Backend Dependencies

### NumPy
- Package: `numpy` (`>=1.26.0`)
- License: BSD-3-Clause
- Project: https://numpy.org/

## Hardware-Dependent Python Components

The app can install PyTorch packages dynamically based on detected hardware (NVIDIA/AMD/CPU).
Because those packages can vary by platform and runtime setup, users should verify the exact installed package set and licenses in their environment before redistribution.

## Dataset and Model Content Notice

This project may be used with external datasets/models obtained by users.
Those assets are **not** covered by this repository's MIT license.
Users are responsible for respecting each dataset/model license and usage terms.

## Disclaimer

This file is a practical notice for open-source distribution and transparency.
It is not legal advice.
