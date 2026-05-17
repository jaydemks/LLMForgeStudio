using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Hardware;

public static class HardwareProbe
{
    public static async Task<HardwareSummary> DetectAsync(string projectRoot, string pythonPath)
    {
        var summary = new HardwareSummary
        {
            LogicalCores = Environment.ProcessorCount,
            OsDescription = RuntimeInformation.OSDescription,
            CpuName = DetectCpuName(),
            TotalRamGb = DetectTotalRamGb(),
            Gpus = DetectGpus()
        };

        summary.BackendDeviceStatus = await DetectBackendDeviceAsync(projectRoot, pythonPath);
        summary.BackendNotes = BuildBackendNotes(summary.Gpus, summary.BackendDeviceStatus);
        return summary;
    }

    private static string DetectCpuName()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var output = RunAndRead("cmd", "/c wmic cpu get Name /value");
                var line = output.Split('\n').FirstOrDefault(x => x.TrimStart().StartsWith("Name=", StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrWhiteSpace(line)) return line.Split('=', 2)[1].Trim();
            }
        }
        catch { }

        return RuntimeInformation.ProcessArchitecture.ToString();
    }

    private static double DetectTotalRamGb()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var output = RunAndRead("cmd", "/c wmic computersystem get TotalPhysicalMemory /value");
                var line = output.Split('\n').FirstOrDefault(x => x.TrimStart().StartsWith("TotalPhysicalMemory=", StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrWhiteSpace(line) && ulong.TryParse(line.Split('=', 2)[1].Trim(), out var bytes))
                    return Math.Round(bytes / 1024d / 1024d / 1024d, 1);
            }
        }
        catch { }

        return 0;
    }

    private static IReadOnlyList<string> DetectGpus()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var output = RunAndRead("cmd", "/c wmic path win32_VideoController get Name");
                var detected = output.Split('\n')
                    .Select(x => x.Trim())
                    .Where(x => !string.IsNullOrWhiteSpace(x) && !x.Equals("Name", StringComparison.OrdinalIgnoreCase))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                // Keep only training-capable physical adapters in primary list.
                var filtered = detected
                    .Where(IsTrainingGpuCandidate)
                    .ToList();

                if (filtered.Count > 0) return filtered;
                if (detected.Count > 0) return new[] { "Nessuna GPU training-capable rilevata (solo adapter virtuali o non supportati)." };
            }
        }
        catch { }

        return new[] { "GPU non rilevata" };
    }

    private static async Task<string> DetectBackendDeviceAsync(string projectRoot, string pythonPath)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(pythonPath))
                return "Python backend non configurato";

            var probeScript = Path.Combine(projectRoot, "backends", "python", "device_probe.py");
            if (!File.Exists(probeScript)) return "Probe backend non disponibile";

            var info = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{probeScript}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(info);
            if (process is null) return "Probe backend non avviabile";

            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            if (process.ExitCode != 0 || string.IsNullOrWhiteSpace(output))
                return "Backend non pronto (torch/device check fallito)";

            using var doc = JsonDocument.Parse(output);
            var root = doc.RootElement;
            var device = root.GetProperty("device").GetString() ?? "unknown";
            var detail = root.TryGetProperty("detail", out var d) ? d.GetString() ?? string.Empty : string.Empty;
            return string.IsNullOrWhiteSpace(detail) ? device : $"{device} - {detail}";
        }
        catch
        {
            return "Backend non pronto (errore probe)";
        }
    }

    private static string BuildBackendNotes(IReadOnlyList<string> gpus, string device)
    {
        var hasNvidia = gpus.Any(g => g.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase));
        var hasAmd = gpus.Any(g => g.Contains("AMD", StringComparison.OrdinalIgnoreCase) || g.Contains("Radeon", StringComparison.OrdinalIgnoreCase));

        if (hasNvidia)
            return "NVIDIA rilevata: training consigliato su CUDA quando disponibile.";

        if (hasAmd)
            return "AMD rilevata: il software funziona ugualmente. Se DirectML/ROCm non è disponibile userà CPU fallback.";

        if (device.Contains("cpu", StringComparison.OrdinalIgnoreCase))
            return "Nessuna accelerazione GPU backend attiva: training su CPU (più lento).";

        return "Configurazione hardware rilevata.";
    }

    private static bool IsTrainingGpuCandidate(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return false;
        if (IsVirtualDisplayAdapter(name)) return false;

        // Keep known training-capable vendors.
        return name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase)
            || name.Contains("GeForce", StringComparison.OrdinalIgnoreCase)
            || name.Contains("RTX", StringComparison.OrdinalIgnoreCase)
            || name.Contains("Quadro", StringComparison.OrdinalIgnoreCase)
            || name.Contains("Tesla", StringComparison.OrdinalIgnoreCase)
            || name.Contains("AMD", StringComparison.OrdinalIgnoreCase)
            || name.Contains("Radeon", StringComparison.OrdinalIgnoreCase)
            || name.Contains("Instinct", StringComparison.OrdinalIgnoreCase)
            || name.Contains("Intel Arc", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsVirtualDisplayAdapter(string name)
    {
        var s = name.Trim();
        return s.Contains("Virtual", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Remote Display", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Basic Render", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Hyper-V", StringComparison.OrdinalIgnoreCase)
            || s.Contains("VMware", StringComparison.OrdinalIgnoreCase)
            || s.Contains("VirtualBox", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Citrix", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Meta Virtual Monitor", StringComparison.OrdinalIgnoreCase)
            || s.Contains("Virtual Desktop Monitor", StringComparison.OrdinalIgnoreCase);
    }

    private static string RunAndRead(string fileName, string arguments)
    {
        var info = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        using var process = Process.Start(info);
        if (process is null) return string.Empty;
        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit(5000);
        return output;
    }
}
