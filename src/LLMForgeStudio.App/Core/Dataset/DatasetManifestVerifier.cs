using System.Text.Json;

namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetManifestVerifier
{
    public static async Task<IReadOnlyList<string>> VerifyAsync(string manifestPath)
    {
        var issues = new List<string>();
        if (!File.Exists(manifestPath))
        {
            issues.Add($"Manifest not found: {manifestPath}");
            return issues;
        }

        var json = await File.ReadAllTextAsync(manifestPath);
        var manifest = JsonSerializer.Deserialize<DatasetShardManifest>(json) ?? new DatasetShardManifest();
        var baseDir = Path.GetDirectoryName(manifestPath) ?? string.Empty;

        var shardItems = manifest.ShardItems.Count > 0
            ? manifest.ShardItems
            : manifest.Shards.Select(path => new DatasetShardItem { RelativePath = path }).ToList();

        for (var i = 0; i < shardItems.Count; i++)
        {
            var shard = shardItems[i];
            var fullPath = Path.IsPathRooted(shard.RelativePath) ? shard.RelativePath : Path.Combine(baseDir, shard.RelativePath);
            if (!File.Exists(fullPath))
            {
                issues.Add($"Missing shard[{i}]: {shard.RelativePath}");
                continue;
            }

            if (!string.IsNullOrWhiteSpace(shard.Sha256))
            {
                var text = await File.ReadAllTextAsync(fullPath);
                var hash = DatasetIntegrity.ComputeSha256FromText(text);
                if (!string.Equals(hash, shard.Sha256, StringComparison.OrdinalIgnoreCase))
                    issues.Add($"Checksum mismatch shard[{i}]: {shard.RelativePath}");
            }
        }

        return issues;
    }
}
