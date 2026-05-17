using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetShardingService
{
    public static async Task<string> BuildManifestFromFolderAsync(string folderPath, string outputDirectory, int targetShardCharacters = 2_000_000)
    {
        Directory.CreateDirectory(outputDirectory);
        var shardsDir = Path.Combine(outputDirectory, "shards");
        Directory.CreateDirectory(shardsDir);

        var files = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly)
            .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        var manifest = new DatasetShardManifest
        {
            RootDirectory = outputDirectory
        };

        var shardIndex = 0;
        var currentChars = 0;
        var current = new StringBuilder();

        async Task FlushShardAsync()
        {
            if (current.Length == 0) return;
            shardIndex++;
            var shardName = $"shard_{shardIndex:0000}.txt";
            var shardPath = Path.Combine(shardsDir, shardName);
            var text = current.ToString();
            await File.WriteAllTextAsync(shardPath, text);
            var relativePath = Path.Combine("shards", shardName);
            var bytes = new FileInfo(shardPath).Length;
            var chars = text.Length;
            var hash = DatasetIntegrity.ComputeSha256FromText(text);
            manifest.Shards.Add(relativePath);
            manifest.ShardItems.Add(new DatasetShardItem
            {
                RelativePath = relativePath,
                Bytes = bytes,
                Characters = chars,
                Sha256 = hash
            });
            manifest.TotalCharacters += chars;
            manifest.TotalBytes += bytes;
            current.Clear();
            currentChars = 0;
        }

        foreach (var file in files)
        {
            var text = await File.ReadAllTextAsync(file);
            if (currentChars > 0 && currentChars + text.Length > targetShardCharacters)
                await FlushShardAsync();

            if (currentChars > 0)
                current.AppendLine().AppendLine();

            current.Append(text);
            currentChars += text.Length;

            if (currentChars >= targetShardCharacters)
                await FlushShardAsync();
        }

        await FlushShardAsync();

        var manifestPath = Path.Combine(outputDirectory, "dataset_manifest.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(manifestPath, json);

        var resumePath = Path.Combine(outputDirectory, "dataset_resume_state.json");
        var resume = new DatasetResumeState
        {
            ManifestPath = manifestPath,
            LastCompletedShardIndex = -1,
            UpdatedAtUtc = DateTimeOffset.UtcNow
        };
        await File.WriteAllTextAsync(resumePath, JsonSerializer.Serialize(resume, new JsonSerializerOptions { WriteIndented = true }));

        return manifestPath;
    }
}
