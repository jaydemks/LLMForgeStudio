using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Export;

public static class OllamaExportPackager
{
    public static bool IsReadyForPackaging(string runDirectory)
    {
        var runDir = Path.GetFullPath(runDirectory);
        var ggufPath = ResolveGgufPath(runDir);
        return !string.IsNullOrWhiteSpace(ggufPath) && File.Exists(ggufPath);
    }

    public static OllamaExportResult CreateFromRun(string runDirectory)
    {
        var runDir = Path.GetFullPath(runDirectory);
        var exportDir = Path.Combine(runDir, "exports", "ollama");
        Directory.CreateDirectory(exportDir);

        var ggufPath = ResolveGgufPath(runDir);
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
        {
            var missing = new OllamaExportResult(
                Status: "blocked",
                ExportDirectory: exportDir,
                ModelFilePath: string.Empty,
                ModelfilePath: string.Empty,
                Notes: "GGUF file not found. Training artifacts were exported, but Ollama bundle cannot be finalized without model.gguf.");
            WriteStatus(exportDir, missing);
            return missing;
        }

        var exportedModelPath = Path.Combine(exportDir, "model.gguf");
        File.Copy(ggufPath, exportedModelPath, overwrite: true);

        var modelName = $"llmforge-{SanitizeName(Path.GetFileName(runDir))}";
        var modelfilePath = Path.Combine(exportDir, "Modelfile");
        var modelfile = new StringBuilder()
            .AppendLine("FROM ./model.gguf")
            .AppendLine("TEMPLATE \"{{ .Prompt }}\"")
            .AppendLine("PARAMETER temperature 0.7")
            .ToString();
        File.WriteAllText(modelfilePath, modelfile, Encoding.UTF8);

        var readmePath = Path.Combine(exportDir, "README_OLLAMA.txt");
        var readme = new StringBuilder()
            .AppendLine("LLMForgeStudio Ollama Export")
            .AppendLine()
            .AppendLine($"Suggested model name: {modelName}")
            .AppendLine()
            .AppendLine("Bundle-only export: this folder is prepared for manual user handoff.")
            .AppendLine("LLMForgeStudio does not write into Ollama internal blobs/manifests storage.")
            .AppendLine("Use only model.gguf + Modelfile from this folder.")
            .ToString();
        File.WriteAllText(readmePath, readme, Encoding.UTF8);

        var success = new OllamaExportResult(
            Status: "ready",
            ExportDirectory: exportDir,
            ModelFilePath: exportedModelPath,
            ModelfilePath: modelfilePath,
            Notes: $"Bundle ready in run folder. Suggested model name: {modelName}.");
        WriteStatus(exportDir, success);
        return success;
    }

    private static string? ResolveGgufPath(string runDir)
    {
        var direct = Path.Combine(runDir, "model.gguf");
        if (File.Exists(direct))
            return direct;

        var manifestPath = Path.Combine(runDir, "checkpoint_manifest.json");
        if (File.Exists(manifestPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath, Encoding.UTF8));
                if (doc.RootElement.TryGetProperty("exportTargets", out var exportTargets)
                    && exportTargets.TryGetProperty("gguf", out var gguf)
                    && gguf.TryGetProperty("status", out var statusEl)
                    && string.Equals(statusEl.GetString(), "exported", StringComparison.OrdinalIgnoreCase)
                    && gguf.TryGetProperty("path", out var pathEl))
                {
                    var manifestValue = pathEl.GetString();
                    if (!string.IsNullOrWhiteSpace(manifestValue))
                    {
                        var resolved = Path.IsPathRooted(manifestValue)
                            ? manifestValue
                            : Path.GetFullPath(Path.Combine(runDir, manifestValue));
                        if (File.Exists(resolved))
                            return resolved;
                    }
                }
            }
            catch
            {
                // Ignore malformed manifest and continue fallback scan.
            }
        }

        var fallback = Directory
            .EnumerateFiles(runDir, "*.gguf", SearchOption.AllDirectories)
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .FirstOrDefault();
        return fallback;
    }

    private static string SanitizeName(string value)
    {
        var chars = value
            .ToLowerInvariant()
            .Select(ch => char.IsLetterOrDigit(ch) ? ch : '-')
            .ToArray();
        var clean = new string(chars);
        while (clean.Contains("--", StringComparison.Ordinal))
            clean = clean.Replace("--", "-", StringComparison.Ordinal);
        return clean.Trim('-');
    }

    private static void WriteStatus(string exportDir, OllamaExportResult result)
    {
        var path = Path.Combine(exportDir, "ollama_export_status.json");
        var payload = new
        {
            result.Status,
            result.ExportDirectory,
            result.ModelFilePath,
            result.ModelfilePath,
            result.Notes,
            generatedAtUtc = DateTimeOffset.UtcNow
        };
        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
    }
}

public sealed record OllamaExportResult(
    string Status,
    string ExportDirectory,
    string ModelFilePath,
    string ModelfilePath,
    string Notes);
