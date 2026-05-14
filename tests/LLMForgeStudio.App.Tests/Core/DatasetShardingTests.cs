using LLMForgeStudio.App.Core.Dataset;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class DatasetShardingTests
{
    [Fact]
    public async Task BuildManifestAndLoad_FromShards_Works()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llmforge-shard-test-{Guid.NewGuid():N}");
        var src = Path.Combine(root, "src");
        var outDir = Path.Combine(root, "out");
        Directory.CreateDirectory(src);

        await File.WriteAllTextAsync(Path.Combine(src, "a.txt"), "hello world");
        await File.WriteAllTextAsync(Path.Combine(src, "b.txt"), "ciao mondo");

        try
        {
            var manifestPath = await DatasetShardingService.BuildManifestFromFolderAsync(src, outDir, targetShardCharacters: 8);
            Assert.True(File.Exists(manifestPath));
            Assert.True(File.Exists(Path.Combine(outDir, "dataset_resume_state.json")));

            var text = await DatasetLoader.LoadTextAsync(manifestPath, new TextCleanerConfig());
            Assert.Contains("hello world", text);
            Assert.Contains("ciao mondo", text);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ManifestVerifier_DetectsTamperedShard()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llmforge-shard-tamper-{Guid.NewGuid():N}");
        var src = Path.Combine(root, "src");
        var outDir = Path.Combine(root, "out");
        Directory.CreateDirectory(src);

        await File.WriteAllTextAsync(Path.Combine(src, "a.txt"), "alpha");
        await File.WriteAllTextAsync(Path.Combine(src, "b.txt"), "beta");

        try
        {
            var manifestPath = await DatasetShardingService.BuildManifestFromFolderAsync(src, outDir, targetShardCharacters: 16);
            var shardPath = Path.Combine(outDir, "shards", "shard_0001.txt");
            await File.WriteAllTextAsync(shardPath, "tampered-data");

            var issues = await DatasetManifestVerifier.VerifyAsync(manifestPath);
            Assert.Contains(issues, x => x.Contains("Checksum mismatch", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
