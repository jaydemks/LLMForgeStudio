using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetLoader
{
    public static async Task<string> LoadTextAsync(string path, TextCleanerConfig config)
    {
        var raw = path.EndsWith("dataset_manifest.json", StringComparison.OrdinalIgnoreCase)
            ? await LoadFromManifestAsync(path)
            : await File.ReadAllTextAsync(path);

        return TextCleaner.Clean(raw, config);
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
