using System.Globalization;
using LLMForgeStudio.App.Core.Tokenization;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private string ResolveGatherWorkspace()
    {
        if (!string.IsNullOrWhiteSpace(GatherWorkspaceDirectory))
            return GatherWorkspaceDirectory;
        var path = Path.Combine(RunDirectory, "gather_dataset");
        GatherWorkspaceDirectory = path;
        return path;
    }

    private enum GatherProviderKind
    {
        HuggingFace,
        GitHub,
        GitLab,
        Zenodo,
        OpenML,
        Uci,
        DataGov,
        GenericHttp,
        LocalFile,
        LocalFolder,
        Unsupported
    }

    private sealed record GatherProviderContract(
        GatherProviderKind Kind,
        string DisplayName,
        bool RequiresExplicitLicenseCheck,
        bool SupportsRemoteLicenseResolver);

    private GatherProviderContract ResolveGatherProviderContract(string src)
    {
        if (string.IsNullOrWhiteSpace(src))
            return new GatherProviderContract(GatherProviderKind.Unsupported, "Unknown", false, false);

        if (TryParseHfDatasetId(src, out _))
            return new GatherProviderContract(GatherProviderKind.HuggingFace, "Hugging Face", true, true);

        if (TryParseGitHubRepoId(src, out _))
            return new GatherProviderContract(GatherProviderKind.GitHub, "GitHub", true, true);

        if (File.Exists(src))
            return new GatherProviderContract(GatherProviderKind.LocalFile, IsEnglish ? "Local file" : "File locale", false, false);

        if (Directory.Exists(src))
            return new GatherProviderContract(GatherProviderKind.LocalFolder, IsEnglish ? "Local folder" : "Cartella locale", false, false);

        if (Uri.TryCreate(src, UriKind.Absolute, out var uri)
            && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps))
        {
            var host = (uri.Host ?? string.Empty).Trim().ToLowerInvariant();
            if (host.Contains("kaggle.com"))
                return new GatherProviderContract(GatherProviderKind.Unsupported, "Kaggle (auth/API required)", false, false);
            if (host.Contains("drive.google.com"))
                return new GatherProviderContract(GatherProviderKind.Unsupported, "Google Drive (auth required)", false, false);
            if (host.Contains("gitlab.com"))
                return new GatherProviderContract(GatherProviderKind.GitLab, "GitLab", false, false);
            if (host.Contains("zenodo.org"))
                return new GatherProviderContract(GatherProviderKind.Zenodo, "Zenodo", false, false);
            if (host.Contains("openml.org"))
                return new GatherProviderContract(GatherProviderKind.OpenML, "OpenML", false, false);
            if (host.Contains("archive.ics.uci.edu"))
                return new GatherProviderContract(GatherProviderKind.Uci, "UCI ML Repository", false, false);
            if (host.Contains("data.gov"))
                return new GatherProviderContract(GatherProviderKind.DataGov, "Data.gov", false, false);

            return new GatherProviderContract(GatherProviderKind.GenericHttp, $"HTTP ({uri.Host})", false, false);
        }

        return new GatherProviderContract(GatherProviderKind.Unsupported, "Unsupported", false, false);
    }

    private void GatherClearSources()
    {
        GatherStagedSources.Clear();
        GatherSourceEntries.Clear();
        GatherStagedDatasetPath = string.Empty;
        GatherValidationText = "-";
        GatherMergeComplianceText = IsEnglish ? "Merge compliance: pending" : "Compliance merge: in attesa";
        GatherStatusText = IsEnglish
            ? "Gather staging cleared. Source files remain on disk."
            : "Staging Gather pulito. I file sorgente restano su disco.";
        OnPropertyChanged(nameof(GatherSourcesCountText));
        OnPropertyChanged(nameof(CanGatherConvertParquet));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(CanGatherMergeSources));
        OnPropertyChanged(nameof(CanGatherValidateDataset));
    }

    private void RemoveGatherSource(GatherSourceEntryViewModel? source)
    {
        if (source is null) return;

        GatherSourceEntries.Remove(source);
        GatherStagedSources.Remove(source.Path);

        if (string.Equals(GatherStagedDatasetPath, source.Path, StringComparison.OrdinalIgnoreCase))
        {
            var next = GatherSourceEntries.FirstOrDefault(x => x.IsEnabled)?.Path
                       ?? GatherSourceEntries.FirstOrDefault()?.Path
                       ?? string.Empty;
            GatherStagedDatasetPath = next;
        }

        GatherStatusText = IsEnglish
            ? $"Source removed: {source.Path}"
            : $"Sorgente rimossa: {source.Path}";

        OnPropertyChanged(nameof(GatherSourcesCountText));
        OnPropertyChanged(nameof(CanGatherConvertParquet));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(CanGatherMergeSources));
        OnPropertyChanged(nameof(CanGatherValidateDataset));
        OnPropertyChanged(nameof(CanGatherHandoffScratch));
        OnPropertyChanged(nameof(CanGatherHandoffFineTune));
    }

    private void AddGatherSource(string path, string providerLabel, string licenseLabel, bool isLicensePermitted)
    {
        if (string.IsNullOrWhiteSpace(path)) return;
        if (!GatherStagedSources.Contains(path))
            GatherStagedSources.Add(path);
        var existing = GatherSourceEntries.FirstOrDefault(x => string.Equals(x.Path, path, StringComparison.OrdinalIgnoreCase));
        if (existing is null)
        {
            var item = new GatherSourceEntryViewModel(path, providerLabel, licenseLabel, isLicensePermitted);
            item.PropertyChanged += (_, _) =>
            {
                OnPropertyChanged(nameof(CanGatherConvertParquet));
                OnPropertyChanged(nameof(CanGatherMergeSources));
                OnPropertyChanged(nameof(CanGatherValidateDataset));
                OnPropertyChanged(nameof(GatherNeedsParquetConversion));
                OnPropertyChanged(nameof(GatherParquetHintText));
            };
            GatherSourceEntries.Add(item);
        }
        else
        {
            existing.IsLicensePermitted = isLicensePermitted;
        }
        OnPropertyChanged(nameof(GatherSourcesCountText));
        OnPropertyChanged(nameof(CanGatherConvertParquet));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(CanGatherMergeSources));
        OnPropertyChanged(nameof(CanGatherValidateDataset));
    }

    private bool DetectGatherParquetInputs()
    {
        try
        {
            static bool PathHasParquet(string p)
            {
                if (string.IsNullOrWhiteSpace(p)) return false;
                if (File.Exists(p))
                    return p.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase);
                if (Directory.Exists(p))
                    return Directory.EnumerateFiles(p, "*.parquet", SearchOption.AllDirectories).Any();
                return false;
            }

            if (GatherSourceEntries.Any(x => x.IsEnabled && PathHasParquet(x.Path)))
                return true;

            return PathHasParquet(GatherStagedDatasetPath);
        }
        catch
        {
            return false;
        }
    }

    private void MarkGatherParquetSourcesConverted(string convertedPath)
    {
        static bool PathHasParquet(string p)
        {
            if (string.IsNullOrWhiteSpace(p)) return false;
            if (File.Exists(p))
                return p.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase);
            if (Directory.Exists(p))
                return Directory.EnumerateFiles(p, "*.parquet", SearchOption.AllDirectories).Any();
            return false;
        }

        foreach (var source in GatherSourceEntries)
        {
            if (string.Equals(source.Path, convertedPath, StringComparison.OrdinalIgnoreCase))
            {
                source.IsEnabled = true;
                continue;
            }

            if (PathHasParquet(source.Path))
                source.IsEnabled = false;
        }

        OnPropertyChanged(nameof(CanGatherConvertParquet));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(CanGatherMergeSources));
        OnPropertyChanged(nameof(CanGatherValidateDataset));
    }

    private string ReplaceGatherParquetSourcesWithConverted(string convertedPath)
    {
        static bool PathHasParquet(string p)
        {
            if (string.IsNullOrWhiteSpace(p)) return false;
            if (File.Exists(p))
                return p.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase);
            if (Directory.Exists(p))
                return Directory.EnumerateFiles(p, "*.parquet", SearchOption.AllDirectories).Any();
            return false;
        }

        static string FriendlyName(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return "source";
            if (File.Exists(path))
                return Path.GetFileNameWithoutExtension(path);
            var trimmed = path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return Path.GetFileName(trimmed);
        }

        var replaced = GatherSourceEntries
            .Where(x => PathHasParquet(x.Path))
            .Select(x => x.Path)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var path in replaced)
        {
            var vm = GatherSourceEntries.FirstOrDefault(x => string.Equals(x.Path, path, StringComparison.OrdinalIgnoreCase));
            if (vm is not null) GatherSourceEntries.Remove(vm);
            GatherStagedSources.Remove(path);
        }

        var existingConverted = GatherSourceEntries.FirstOrDefault(x => string.Equals(x.Path, convertedPath, StringComparison.OrdinalIgnoreCase));
        if (existingConverted is not null)
            GatherSourceEntries.Remove(existingConverted);
        GatherStagedSources.Remove(convertedPath);

        var names = replaced.Select(FriendlyName).Where(x => !string.IsNullOrWhiteSpace(x)).Take(2).ToList();
        var tail = replaced.Count > 2 ? " +" + (replaced.Count - 2).ToString(CultureInfo.InvariantCulture) : string.Empty;
        var fromText = names.Count > 0 ? string.Join(" + ", names) + tail : "parquet source";
        var label = IsEnglish ? $"Converted from {fromText}" : $"Convertito da {fromText}";
        return label;
    }

    private void GatherApplyRecommendations()
    {
        if (string.Equals(GatherRecommendedTokenizer, "ByteLevelBpe", StringComparison.OrdinalIgnoreCase))
            SelectedTokenizerKind = TokenizerKind.ByteLevelBpe;
        else if (string.Equals(GatherRecommendedTokenizer, "SimpleBpe", StringComparison.OrdinalIgnoreCase))
            SelectedTokenizerKind = TokenizerKind.SimpleBpe;

        var targetProfile = GatherRecommendedTrainingProfile;
        if (string.IsNullOrWhiteSpace(targetProfile) || !TrainingProfileOptions.Contains(targetProfile))
            targetProfile = "Balanced";

        // Hardware-aware adaptation layer on top of dataset recommendation.
        if (TrainingForceCpu)
        {
            if (string.Equals(targetProfile, "Cluster", StringComparison.OrdinalIgnoreCase)
                || string.Equals(targetProfile, "Research", StringComparison.OrdinalIgnoreCase)
                || string.Equals(targetProfile, "Serious", StringComparison.OrdinalIgnoreCase))
                targetProfile = "Balanced";
            TrainingMixedPrecision = false;
            TrainingDistributed = false;
            TrainingMultiGpuStrategy = "none";
        }
        else if (_lastHardwareSummary is not null)
        {
            var hasAcceleratedGpu = _lastHardwareSummary.Gpus.Any(g =>
                !g.Contains("non rilevata", StringComparison.OrdinalIgnoreCase)
                && !g.Contains("not detected", StringComparison.OrdinalIgnoreCase));

            if (!hasAcceleratedGpu &&
                (string.Equals(targetProfile, "Cluster", StringComparison.OrdinalIgnoreCase)
                 || string.Equals(targetProfile, "Research", StringComparison.OrdinalIgnoreCase)))
            {
                targetProfile = "Serious";
            }

            if (_lastHardwareSummary.TotalRamGb > 0 && _lastHardwareSummary.TotalRamGb < 24
                && string.Equals(targetProfile, "Serious", StringComparison.OrdinalIgnoreCase))
            {
                targetProfile = "Balanced";
            }
        }

        SelectedTrainingProfile = targetProfile;

        SelectedSection = "Tokenization";
        GatherStatusText = $"Recommendations applied ({targetProfile}). Jumped to Tokenization.";
    }
}
