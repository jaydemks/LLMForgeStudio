using System.Diagnostics;
using System.Text.Json;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.Core.Backend;

public static class PythonBackendBridge
{
    public static async Task<string> WriteJobSpecAsync(BackendJobSpec spec, string directory)
    {
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "job_spec.json");
        var json = JsonSerializer.Serialize(spec, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json);
        return path;
    }

    public static bool IsPythonAvailable(string pythonExe)
    {
        try
        {
            var info = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = "--version",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var process = Process.Start(info);
            if (process is null) return false;
            process.WaitForExit(5000);
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    public static ProcessStartInfo CreateStartInfo(string pythonExe, string scriptPath, string arguments)
    {
        return new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"\"{scriptPath}\" {arguments}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
    }

    public static async Task<TrainingLogEntry?> TryReadLogEntryAsync(string logPath)
    {
        if (!File.Exists(logPath)) return null;
        var lines = await File.ReadAllLinesAsync(logPath);
        var last = lines.LastOrDefault(l => !string.IsNullOrWhiteSpace(l));
        if (string.IsNullOrWhiteSpace(last)) return null;

        using var doc = JsonDocument.Parse(last);
        var root = doc.RootElement;
        return new TrainingLogEntry(
            root.GetProperty("step").GetInt32(),
            root.GetProperty("train_loss").GetDouble(),
            root.GetProperty("val_loss").GetDouble(),
            root.GetProperty("tokens_per_second").GetDouble(),
            root.TryGetProperty("message", out var message) ? message.GetString() ?? string.Empty : string.Empty
        );
    }
}
