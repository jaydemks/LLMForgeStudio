using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Alignment;

public static class SftDatasetFormatter
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

            if (!TryFormatJsonlRecord(trimmed, out var formatted))
            {
                formatted = "<|user|>\n" + trimmed + "\n<|assistant|>\n";
            }

            if (sb.Length > 0) sb.AppendLine().AppendLine();
            sb.Append(formatted.TrimEnd());
        }

        return sb.ToString();
    }

    private static bool TryFormatJsonlRecord(string jsonLine, out string formatted)
    {
        formatted = string.Empty;
        try
        {
            using var doc = JsonDocument.Parse(jsonLine);
            var root = doc.RootElement;

            if (root.TryGetProperty("prompt", out var prompt) && root.TryGetProperty("response", out var response))
            {
                formatted = $"<|user|>\n{prompt.GetString() ?? string.Empty}\n<|assistant|>\n{response.GetString() ?? string.Empty}";
                return true;
            }

            if (root.TryGetProperty("messages", out var messages) && messages.ValueKind == JsonValueKind.Array)
            {
                var sb = new StringBuilder();
                foreach (var m in messages.EnumerateArray())
                {
                    var role = m.TryGetProperty("role", out var r) ? (r.GetString() ?? string.Empty) : string.Empty;
                    var content = m.TryGetProperty("content", out var c) ? (c.GetString() ?? string.Empty) : string.Empty;
                    var token = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? "assistant" : "user";
                    if (sb.Length > 0) sb.AppendLine();
                    sb.Append("<|").Append(token).Append("|>\n").Append(content);
                }

                if (sb.Length > 0)
                {
                    formatted = sb.ToString();
                    return true;
                }
            }
        }
        catch
        {
            // non-json line
        }

        return false;
    }
}
