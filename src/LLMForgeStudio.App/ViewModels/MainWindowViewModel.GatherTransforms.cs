using System.Text.Json;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private static readonly string[] MetadataJsonKeys =
    {
        "_data_files", "_fingerprint", "_format_columns", "_format_kwargs", "_format_type", "_output_all_columns", "_split",
        "dataset_info", "features", "license", "homepage", "citation", "description", "task_categories", "size_categories"
    };

    private static string ExtractLicenseLabelFromStatus(string statusText)
    {
        if (string.IsNullOrWhiteSpace(statusText)) return "unknown";
        var idx = statusText.IndexOf(':');
        if (idx < 0 || idx >= statusText.Length - 1) return "unknown";
        var right = statusText[(idx + 1)..].Trim();
        var end = right.IndexOf(' ');
        return end > 0 ? right[..end].Trim() : right;
    }

    private static string ApplyGatherDedupPolicy(string input, string policy)
    {
        if (string.IsNullOrWhiteSpace(input)) return string.Empty;
        var normalizedPolicy = (policy ?? "line").Trim().ToLowerInvariant();
        if (normalizedPolicy == "none") return input;

        if (normalizedPolicy == "paragraph")
        {
            var parts = input.Split(new[] { "\r\n\r\n", "\n\n" }, StringSplitOptions.RemoveEmptyEntries)
                .Select(x => x.Trim())
                .Where(x => !string.IsNullOrWhiteSpace(x));
            return string.Join("\n\n", parts.Distinct());
        }

        var lines = input.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(x => x.Trim())
            .Where(x => !string.IsNullOrWhiteSpace(x));
        return string.Join('\n', lines.Distinct());
    }

    private static async Task<List<string>> ExtractNormalizedTrainingTextsAsync(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".jsonl")
            return await ExtractFromJsonlAsync(path);
        if (ext == ".json")
            return await ExtractFromJsonAsync(path);
        if (ext == ".csv")
            return await ExtractFromCsvAsync(path);

        var raw = await File.ReadAllTextAsync(path);
        return string.IsNullOrWhiteSpace(raw) ? new List<string>() : new List<string> { raw };
    }

    private static async Task<List<string>> ExtractFromJsonlAsync(string path)
    {
        return await Task.Run(async () =>
        {
            var outTexts = new List<string>();
            foreach (var line in await File.ReadAllLinesAsync(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var extracted = ExtractTextFromJsonElement(doc.RootElement);
                    if (!string.IsNullOrWhiteSpace(extracted)) outTexts.Add(extracted);
                }
                catch
                {
                    outTexts.Add(line);
                }
            }
            return outTexts;
        });
    }

    private static async Task<List<string>> ExtractFromJsonAsync(string path)
    {
        return await Task.Run(async () =>
        {
            var raw = await File.ReadAllTextAsync(path);
            if (string.IsNullOrWhiteSpace(raw)) return new List<string>();
            try
            {
                using var doc = JsonDocument.Parse(raw);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    var arr = new List<string>();
                    foreach (var el in doc.RootElement.EnumerateArray())
                    {
                        var t = ExtractTextFromJsonElement(el);
                        if (!string.IsNullOrWhiteSpace(t)) arr.Add(t);
                    }
                    return arr;
                }
                if (LooksLikeMetadataOnlyJson(doc.RootElement))
                    return new List<string>();
                var one = ExtractTextFromJsonElement(doc.RootElement);
                return string.IsNullOrWhiteSpace(one) ? new List<string>() : new List<string> { one };
            }
            catch
            {
                return new List<string> { raw };
            }
        });
    }

    private static async Task<List<string>> ExtractFromCsvAsync(string path)
    {
        return await Task.Run(async () =>
        {
            var lines = await File.ReadAllLinesAsync(path);
            var outTexts = new List<string>();
            foreach (var line in lines.Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var cols = line.Split(',', StringSplitOptions.TrimEntries);
                if (cols.Length == 0) continue;
                outTexts.Add(string.Join(" ", cols));
            }
            return outTexts;
        });
    }

    private static bool LooksLikeMetadataOnlyJson(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object) return false;

        var hasTrainingFields =
            root.TryGetProperty("text", out var textEl) && textEl.ValueKind == JsonValueKind.String
            || root.TryGetProperty("messages", out var messagesEl) && messagesEl.ValueKind == JsonValueKind.Array
            || root.TryGetProperty("prompt", out _) && root.TryGetProperty("response", out _)
            || root.TryGetProperty("conversations", out _);
        if (hasTrainingFields) return false;

        var propNames = root.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var metadataHits = MetadataJsonKeys.Count(k => propNames.Contains(k));
        return metadataHits >= 2;
    }
}
