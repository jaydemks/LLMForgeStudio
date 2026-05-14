using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Alignment;

public static class DpoDatasetFormatter
{
    public static string FormatToTrainingText(string rawText)
    {
        if (string.IsNullOrWhiteSpace(rawText)) return string.Empty;

        var lines = rawText.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var sb = new StringBuilder();

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0) continue;

            if (!TryFormatPreferenceRecord(trimmed, out var formatted))
                continue;

            if (sb.Length > 0) sb.AppendLine().AppendLine();
            sb.Append(formatted.TrimEnd());
        }

        return sb.ToString();
    }

    private static bool TryFormatPreferenceRecord(string jsonLine, out string formatted)
    {
        formatted = string.Empty;
        try
        {
            using var doc = JsonDocument.Parse(jsonLine);
            var root = doc.RootElement;

            var prompt = root.TryGetProperty("prompt", out var p) ? p.GetString() ?? string.Empty : string.Empty;
            var chosen = root.TryGetProperty("chosen", out var c) ? c.GetString() ?? string.Empty : string.Empty;
            var rejected = root.TryGetProperty("rejected", out var r) ? r.GetString() ?? string.Empty : string.Empty;

            if (string.IsNullOrWhiteSpace(prompt) || string.IsNullOrWhiteSpace(chosen) || string.IsNullOrWhiteSpace(rejected))
                return false;

            formatted = string.Join("\n",
                "<|user|>",
                prompt,
                "<|assistant_chosen|>",
                chosen,
                "<|assistant_rejected|>",
                rejected);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
