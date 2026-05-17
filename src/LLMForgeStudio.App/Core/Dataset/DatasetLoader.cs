using System.Text;
using System.Text.Json;
using System.Linq;

namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetLoader
{
    private static readonly string[] MetadataJsonKeys =
    {
        "_data_files", "_fingerprint", "_format_columns", "_format_kwargs", "_format_type", "_output_all_columns", "_split",
        "dataset_info", "features", "license", "homepage", "citation", "description", "task_categories", "size_categories"
    };

    public static async Task<string> LoadTextAsync(string path, TextCleanerConfig config)
    {
        var raw = path.EndsWith("dataset_manifest.json", StringComparison.OrdinalIgnoreCase)
            ? await LoadFromManifestAsync(path)
            : await LoadStructuredTextAsync(path);

        return TextCleaner.Clean(raw, config);
    }

    private static async Task<string> LoadStructuredTextAsync(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".jsonl")
            return await LoadFromJsonlAsync(path);
        if (ext == ".json")
            return await LoadFromJsonAsync(path);
        if (ext == ".csv")
            return await LoadFromCsvAsync(path);
        return await File.ReadAllTextAsync(path);
    }

    private static async Task<string> LoadFromJsonlAsync(string path)
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

        return string.Join('\n', outTexts);
    }

    private static async Task<string> LoadFromJsonAsync(string path)
    {
        var raw = await File.ReadAllTextAsync(path);
        if (string.IsNullOrWhiteSpace(raw)) return string.Empty;

        try
        {
            using var doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind == JsonValueKind.Array)
            {
                var outTexts = new List<string>();
                foreach (var el in doc.RootElement.EnumerateArray())
                {
                    var extracted = ExtractTextFromJsonElement(el);
                    if (!string.IsNullOrWhiteSpace(extracted)) outTexts.Add(extracted);
                }
                return string.Join('\n', outTexts);
            }

            if (LooksLikeMetadataOnlyJson(doc.RootElement))
                return string.Empty;

            var single = ExtractTextFromJsonElement(doc.RootElement);
            return string.IsNullOrWhiteSpace(single) ? string.Empty : single;
        }
        catch
        {
            return raw;
        }
    }

    private static async Task<string> LoadFromCsvAsync(string path)
    {
        var lines = await File.ReadAllLinesAsync(path);
        if (lines.Length <= 1) return string.Empty;
        var outTexts = new List<string>();
        foreach (var line in lines.Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var cols = line.Split(',', StringSplitOptions.TrimEntries);
            if (cols.Length == 0) continue;
            outTexts.Add(string.Join(" ", cols));
        }
        return string.Join('\n', outTexts);
    }

    private static string ExtractTextFromJsonElement(JsonElement root)
    {
        if (root.ValueKind == JsonValueKind.Object)
        {
            if (root.TryGetProperty("prompt", out var prompt) && root.TryGetProperty("response", out var response))
                return $"<|user|>\n{prompt.GetString() ?? string.Empty}\n<|assistant|>\n{response.GetString() ?? string.Empty}".Trim();

            if (root.TryGetProperty("messages", out var messages) && messages.ValueKind == JsonValueKind.Array)
            {
                var sb = new StringBuilder();
                foreach (var m in messages.EnumerateArray())
                {
                    var role = m.TryGetProperty("role", out var r) ? r.GetString() : "user";
                    var content = m.TryGetProperty("content", out var c) ? c.GetString() : string.Empty;
                    if (string.IsNullOrWhiteSpace(content)) continue;
                    if (sb.Length > 0) sb.Append('\n');
                    if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase))
                        sb.Append("<|assistant|>\n").Append(content);
                    else
                        sb.Append("<|user|>\n").Append(content);
                }
                return sb.ToString().Trim();
            }

            if (root.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
                return text.GetString() ?? string.Empty;
        }

        return root.ToString();
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

    private static async Task<string> LoadFromManifestAsync(string manifestPath)
    {
        var issues = await DatasetManifestVerifier.VerifyAsync(manifestPath);
        if (issues.Count > 0)
            throw new InvalidOperationException($"Dataset manifest integrity failed: {string.Join(" | ", issues)}");

        var json = await File.ReadAllTextAsync(manifestPath);
        var manifest = JsonSerializer.Deserialize<DatasetShardManifest>(json) ?? new DatasetShardManifest();
        var baseDir = Path.GetDirectoryName(manifestPath) ?? string.Empty;

        var shards = manifest.ShardItems.Count > 0
            ? manifest.ShardItems.Select(s => s.RelativePath).ToList()
            : manifest.Shards;

        var sb = new StringBuilder();
        foreach (var shard in shards)
        {
            var fullPath = Path.IsPathRooted(shard) ? shard : Path.Combine(baseDir, shard);
            if (!File.Exists(fullPath)) continue;

            var text = await File.ReadAllTextAsync(fullPath);
            if (sb.Length > 0) sb.AppendLine().AppendLine();
            sb.Append(text);
        }

        return sb.ToString();
    }
}
