using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LLMForgeStudio.App.Core.Backend;
using LLMForgeStudio.App.Core.Dataset;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private void CancelGatherOperation()
    {
        _gatherCts?.Cancel();
    }

    private async Task GatherFetchSourceAsync()
    {
        if (GatherIsBusy) return;
        _gatherCts?.Cancel();
        _gatherCts?.Dispose();
        _gatherCts = new CancellationTokenSource();
        var cancellationToken = _gatherCts.Token;
        var sw = Stopwatch.StartNew();
        GatherIsBusy = true;
        GatherProgressValue = 5;
        GatherProgressText = "Fetching source...";
        var src = (GatherSourceInput ?? string.Empty).Trim();
        var provider = ResolveGatherProviderContract(src);
        GatherSourceProviderText = $"Provider: {provider.DisplayName}";
        var existingStagedPath = TryFindExistingStagedSourcePath(src, provider);
        if (!string.IsNullOrWhiteSpace(existingStagedPath))
        {
            GatherStatusText = $"Source already staged: {existingStagedPath}";
            GatherProgressValue = 100;
            GatherProgressText = "Fetch completed.";
            GatherIsBusy = false;
            return;
        }
        if (string.IsNullOrWhiteSpace(src))
        {
            GatherStatusText = "Source required (URL, file or folder).";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }
        if (provider.Kind == GatherProviderKind.Unsupported)
        {
            GatherStatusText = IsEnglish
                ? "Unsupported provider. Use Hugging Face/GitHub dataset links, direct HTTP file URL, local file, or local folder."
                : "Provider non supportato. Usa link dataset Hugging Face/GitHub, URL file HTTP diretto, file locale o cartella locale.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        if (provider.RequiresExplicitLicenseCheck)
        {
            if (!GatherLicensePermitted)
            {
                GatherStatusText = "Blocked: license is not approved for unrestricted use.";
                GatherIsBusy = false;
                GatherProgressValue = 0;
                GatherProgressText = "Idle";
                return;
            }
            if (!GatherLicenseAcknowledged)
            {
                GatherStatusText = "Blocked: acknowledge license compliance first.";
                GatherIsBusy = false;
                GatherProgressValue = 0;
                GatherProgressText = "Idle";
                return;
            }
        }
        else if (!GatherLicenseAcknowledged)
        {
            GatherStatusText = "Blocked: confirm you have rights to use this source.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var workspace = ResolveGatherWorkspace();
        var sourcesDir = Path.Combine(workspace, "sources");
        Directory.CreateDirectory(sourcesDir);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Uri.TryCreate(src, UriKind.Absolute, out var uri)
                && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps))
            {
                var ext = Path.GetExtension(uri.AbsolutePath ?? string.Empty).ToLowerInvariant();
                var directDatasetUrl = IsSupportedDatasetExtension(ext) || string.Equals(ext, ".parquet", StringComparison.OrdinalIgnoreCase);
                var providerPageLike = provider.Kind is GatherProviderKind.HuggingFace or GatherProviderKind.GitHub;
                if (providerPageLike && !directDatasetUrl)
                {
                    var handled = await TryFetchProviderRepositorySnapshotAsync(src, provider, sourcesDir, sw, cancellationToken);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (handled) return;
                }

                using var http = new HttpClient();
                var fileName = Path.GetFileName(uri.LocalPath);
                if (string.IsNullOrWhiteSpace(fileName)) fileName = "downloaded_dataset.bin";
                var outPath = EnsureUniquePath(Path.Combine(sourcesDir, fileName), isDirectory: false);
                using var response = await http.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();
                await using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.None);
                await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
                var totalBytesExpected = response.Content.Headers.ContentLength.GetValueOrDefault(-1);
                var downloadedBytes = 0L;
                var buffer = new byte[128 * 1024];
                var lastUiUpdate = Stopwatch.StartNew();
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var read = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken);
                    if (read <= 0) break;
                    await fs.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                    downloadedBytes += read;
                    if (lastUiUpdate.ElapsedMilliseconds >= 200)
                    {
                        var pct = totalBytesExpected > 0
                            ? Math.Clamp((downloadedBytes / (double)totalBytesExpected) * 100.0, 0, 99.0)
                            : 0.0;
                        GatherProgressValue = totalBytesExpected > 0 ? pct : 15;
                        var downloadedMb = downloadedBytes / (1024.0 * 1024.0);
                        var totalMb = totalBytesExpected > 0 ? totalBytesExpected / (1024.0 * 1024.0) : 0.0;
                        var speedMbPerSec = downloadedMb / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                        GatherProgressText = totalBytesExpected > 0
                            ? $"Fetching source... {downloadedMb:F1}/{totalMb:F1} MB | {speedMbPerSec:F2} MB/s | elapsed {sw.Elapsed:mm\\:ss}"
                            : $"Fetching source... {downloadedMb:F1} MB | {speedMbPerSec:F2} MB/s | elapsed {sw.Elapsed:mm\\:ss}";
                        lastUiUpdate.Restart();
                    }
                }
                var finalDownloadedBytes = new FileInfo(outPath).Length;
                GatherProgressValue = 100;
                GatherStagedDatasetPath = outPath;
                var licenseLabel = provider.Kind == GatherProviderKind.HuggingFace
                    ? ExtractLicenseLabelFromStatus(GatherLicenseText)
                    : "manual-ack";
                AddGatherSource(outPath, provider.DisplayName, licenseLabel, provider.Kind == GatherProviderKind.HuggingFace ? GatherLicensePermitted : true);
                GatherStatusText = $"Source downloaded: {outPath} | size={finalDownloadedBytes:N0} bytes | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                GatherProgressText = "Fetch completed.";
                GatherIsBusy = false;
                return;
            }

            if (File.Exists(src))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var dest = EnsureUniquePath(Path.Combine(sourcesDir, Path.GetFileName(src)), isDirectory: false);
                File.Copy(src, dest, overwrite: true);
                var stagedBytes = new FileInfo(dest).Length;
                GatherStagedDatasetPath = dest;
                var licenseLabel = provider.Kind == GatherProviderKind.HuggingFace
                    ? ExtractLicenseLabelFromStatus(GatherLicenseText)
                    : "manual-ack";
                AddGatherSource(dest, provider.DisplayName, licenseLabel, provider.Kind == GatherProviderKind.HuggingFace ? GatherLicensePermitted : true);
                GatherStatusText = $"Source file staged: {dest} | size={stagedBytes:N0} bytes | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                GatherProgressValue = 100;
                GatherProgressText = "Fetch completed.";
                GatherIsBusy = false;
                return;
            }

            if (Directory.Exists(src))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var baseDir = Path.Combine(sourcesDir, Path.GetFileName(src.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)));
                var destDir = EnsureUniquePath(baseDir, isDirectory: true);
                Directory.CreateDirectory(destDir);
                var files = Directory.EnumerateFiles(src, "*.*", SearchOption.AllDirectories).ToList();
                var total = Math.Max(1, files.Count);
                var idx = 0;
                foreach (var file in files)
                {
                    var rel = Path.GetRelativePath(src, file);
                    var outFile = Path.Combine(destDir, rel);
                    Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? destDir);
                    File.Copy(file, outFile, overwrite: true);
                    idx++;
                    GatherProgressValue = 5 + (idx / (double)total) * 95.0;
                    if (idx % 25 == 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        await Task.Yield();
                    }
                }
                GatherStagedDatasetPath = destDir;
                var stagedFiles = Directory.EnumerateFiles(destDir, "*.*", SearchOption.AllDirectories).Count();
                var licenseLabel = provider.Kind == GatherProviderKind.HuggingFace
                    ? ExtractLicenseLabelFromStatus(GatherLicenseText)
                    : "manual-ack";
                AddGatherSource(destDir, provider.DisplayName, licenseLabel, provider.Kind == GatherProviderKind.HuggingFace ? GatherLicensePermitted : true);
                GatherStatusText = $"Source folder staged: {destDir} | files={stagedFiles:N0} | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                GatherProgressText = "Fetch completed.";
                GatherIsBusy = false;
                return;
            }

            GatherStatusText = "Source not found.";
        }
        catch (OperationCanceledException)
        {
            GatherStatusText = IsEnglish ? "Fetch canceled by user." : "Fetch annullato dall'utente.";
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
        }
        catch (Exception ex)
        {
            GatherStatusText = $"Fetch failed: {ex.Message}";
        }
        finally
        {
            GatherIsBusy = false;
            _gatherCts?.Dispose();
            _gatherCts = null;
            if (GatherProgressValue < 100) GatherProgressValue = 0;
            if (GatherProgressText != "Fetch completed.") GatherProgressText = "Idle";
        }
    }

    private string TryFindExistingStagedSourcePath(string src, GatherProviderContract provider)
    {
        if (string.IsNullOrWhiteSpace(src)) return string.Empty;
        try
        {
            if (provider.Kind == GatherProviderKind.HuggingFace && TryParseHfDatasetId(src, out var datasetId))
            {
                var token = $"hf_{SanitizePathToken(datasetId)}";
                var hit = GatherSourceEntries.FirstOrDefault(x => x.Path.Contains(token, StringComparison.OrdinalIgnoreCase));
                if (hit is not null && (File.Exists(hit.Path) || Directory.Exists(hit.Path)))
                    return hit.Path;
            }
            if (provider.Kind == GatherProviderKind.GitHub && TryParseGitHubRepoId(src, out var repoId))
            {
                var token = $"gh_{SanitizePathToken(repoId)}";
                var hit = GatherSourceEntries.FirstOrDefault(x => x.Path.Contains(token, StringComparison.OrdinalIgnoreCase));
                if (hit is not null && (File.Exists(hit.Path) || Directory.Exists(hit.Path)))
                    return hit.Path;
            }
            if (provider.Kind == GatherProviderKind.LocalFile)
            {
                var fileName = Path.GetFileName(src);
                var hit = GatherSourceEntries.FirstOrDefault(x =>
                    string.Equals(Path.GetFileName(x.Path), fileName, StringComparison.OrdinalIgnoreCase));
                if (hit is not null && File.Exists(hit.Path))
                    return hit.Path;
            }
            if (provider.Kind == GatherProviderKind.LocalFolder)
            {
                var folderName = Path.GetFileName(src.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                var hit = GatherSourceEntries.FirstOrDefault(x =>
                    string.Equals(Path.GetFileName(x.Path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)), folderName, StringComparison.OrdinalIgnoreCase));
                if (hit is not null && Directory.Exists(hit.Path))
                    return hit.Path;
            }
        }
        catch
        {
            // ignore and fallback to fetch
        }
        return string.Empty;
    }

    private async Task GatherCheckLicenseAsync()
    {
        var src = (GatherSourceInput ?? string.Empty).Trim();
        var provider = ResolveGatherProviderContract(src);
        GatherSourceProviderText = $"Provider: {provider.DisplayName}";
        if (provider.Kind == GatherProviderKind.Unsupported)
        {
            GatherLicensePermitted = false;
            GatherLicenseText = IsEnglish
                ? "Unsupported provider: automatic compliance check unavailable."
                : "Provider non supportato: controllo compliance automatico non disponibile.";
            OnPropertyChanged(nameof(CanGatherFetchSource));
            return;
        }

        if (!provider.SupportsRemoteLicenseResolver)
        {
            _gatherHfDatasetId = string.Empty;
            GatherLicensePermitted = true;
            GatherLicenseText = IsEnglish
                ? "No remote license API for this provider. Manual rights confirmation required."
                : "Nessuna API licenza remota per questo provider. Conferma manuale dei diritti richiesta.";
            OnPropertyChanged(nameof(CanGatherFetchSource));
            return;
        }

        await ResolveRemoteProviderLicenseAsync(src, provider);
        OnPropertyChanged(nameof(CanGatherFetchSource));
    }

    private async Task GatherConvertParquetAsync()
    {
        if (GatherIsBusy) return;
        GatherIsBusy = true;
        GatherProgressValue = 10;
        GatherProgressText = "Converting parquet...";
        var staged = GatherStagedDatasetPath;
        if (string.IsNullOrWhiteSpace(staged))
        {
            GatherStatusText = "No staged source. Fetch source first.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var inputIsParquet = (File.Exists(staged) && staged.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
            || (Directory.Exists(staged) && Directory.EnumerateFiles(staged, "*.parquet", SearchOption.AllDirectories).Any());
        if (!inputIsParquet)
        {
            GatherStatusText = "No parquet conversion needed.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var outputDir = Path.Combine(ResolveGatherWorkspace(), "converted");
        Directory.CreateDirectory(outputDir);
        var projectRoot = ResolveProjectRoot();
        var script = Path.Combine(projectRoot, "backends", "python", "parquet_to_jsonl.py");
        var startInfo = PythonBackendBridge.CreateStartInfo(PythonPath, script, $"--input \"{staged}\" --output \"{outputDir}\"");
        using var process = Process.Start(startInfo);
        if (process is null)
        {
            GatherStatusText = "Parquet conversion failed to start.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }
        var stdOut = await process.StandardOutput.ReadToEndAsync();
        var stdErr = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        if (process.ExitCode != 0)
        {
            GatherStatusText = $"Parquet conversion failed ({process.ExitCode}).";
            Log = string.IsNullOrWhiteSpace(stdErr) ? stdOut : stdErr;
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }
        GatherProgressValue = 100;
        GatherStagedDatasetPath = outputDir;
        var convertedLabel = ReplaceGatherParquetSourcesWithConverted(outputDir);
        AddGatherSource(outputDir, convertedLabel, "derived", true);
        MarkGatherParquetSourcesConverted(outputDir);
        GatherStatusText = "Parquet conversion completed.";
        GatherProgressText = "Conversion completed.";
        GatherIsBusy = false;
    }

    private async Task GatherMergeSourcesAsync()
    {
        if (GatherIsBusy) return;
        var sw = Stopwatch.StartNew();
        GatherIsBusy = true;
        GatherProgressValue = 5;
        GatherProgressText = "Merging sources...";
        if (GatherSourceEntries.Count == 0)
        {
            GatherStatusText = "No sources to merge.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var enabledSources = GatherSourceEntries.Where(x => x.IsEnabled).ToList();
        if (enabledSources.Count < 1)
        {
            GatherStatusText = IsEnglish ? "Enable at least 1 source for merge." : "Abilita almeno 1 sorgente per il merge.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var workspace = ResolveGatherWorkspace();
        var mergedDir = Path.Combine(workspace, "merged");
        Directory.CreateDirectory(mergedDir);
        var complianceReportPath = Path.Combine(mergedDir, "dataset_merge_compliance_report.json");

        var blockedLicenses = enabledSources.Where(x => !x.IsLicensePermitted).ToList();
        if (blockedLicenses.Count > 0)
        {
            GatherMergeComplianceText = IsEnglish
                ? $"Merge blocked: {blockedLicenses.Count} source(s) have restricted/unknown license."
                : $"Merge bloccato: {blockedLicenses.Count} sorgente/i con licenza limitata/sconosciuta.";
            GatherStatusText = GatherMergeComplianceText;
            var blockedReport = new
            {
                generatedAtUtc = DateTimeOffset.UtcNow,
                status = "blocked",
                reason = "restricted-or-unknown-license",
                enabledSources = enabledSources.Select(s => new { s.Path, s.Provider, s.LicenseLabel, s.IsLicensePermitted }),
            };
            await File.WriteAllTextAsync(complianceReportPath, JsonSerializer.Serialize(blockedReport, new JsonSerializerOptions { WriteIndented = true }));
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var (matrixPass, matrixReason, matrixChecks, normalizedSources) = EvaluateStrictMergeLicenseMatrix(enabledSources);
        if (!matrixPass)
        {
            GatherMergeComplianceText = IsEnglish
                ? $"Merge blocked: {matrixReason}"
                : $"Merge bloccato: {matrixReason}";
            GatherStatusText = GatherMergeComplianceText;
            var matrixBlockedReport = new
            {
                generatedAtUtc = DateTimeOffset.UtcNow,
                status = "blocked",
                reason = matrixReason,
                enabledSources = normalizedSources,
                pairChecks = matrixChecks
            };
            await File.WriteAllTextAsync(complianceReportPath, JsonSerializer.Serialize(matrixBlockedReport, new JsonSerializerOptions { WriteIndented = true }));
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }
        GatherMergeComplianceText = IsEnglish ? "Merge compliance: passed (strict matrix)." : "Compliance merge: ok (matrice rigorosa).";

        var mergedPath = Path.Combine(mergedDir, "dataset_merged.txt");
        var mergeMetaPath = Path.Combine(mergedDir, "dataset_merged_provenance.json");

        var chunks = new List<string>();
        var provenance = new List<object>();
        var sourcesWithNoReadableData = new List<string>();
        var totalFilesScanned = 0;
        var totalRecordsRead = 0;
        var totalSources = Math.Max(1, enabledSources.Count);
        var sourceIndex = 0;
        foreach (var sourceEntry in enabledSources)
        {
            var source = sourceEntry.Path;
            var localChunks = new List<string>();
            if (File.Exists(source))
            {
                if (IsSupportedDatasetFile(source))
                {
                    totalFilesScanned++;
                    localChunks.AddRange(await ExtractNormalizedTrainingTextsAsync(source));
                }
                sourceIndex++;
                GatherProgressValue = 5 + (sourceIndex / (double)totalSources) * 90.0;
            }
            else if (Directory.Exists(source))
            {
                foreach (var f in Directory.EnumerateFiles(source, "*.*", SearchOption.AllDirectories).Where(IsSupportedDatasetFile))
                {
                    totalFilesScanned++;
                    localChunks.AddRange(await ExtractNormalizedTrainingTextsAsync(f));
                }
                sourceIndex++;
                GatherProgressValue = 5 + (sourceIndex / (double)totalSources) * 90.0;
            }

            if (localChunks.Count > 0)
            {
                totalRecordsRead += localChunks.Count;
                var weighted = string.Join("\n\n", localChunks);
                for (var i = 0; i < sourceEntry.Weight; i++)
                    chunks.Add(weighted);

                provenance.Add(new
                {
                    source = sourceEntry.Path,
                    provider = sourceEntry.Provider,
                    license = sourceEntry.LicenseLabel,
                    weight = sourceEntry.Weight,
                    recordsRead = localChunks.Count,
                    charactersBeforeWeight = weighted.Length,
                    charactersAfterWeight = weighted.Length * sourceEntry.Weight
                });
            }
            else
            {
                sourcesWithNoReadableData.Add(sourceEntry.Path);
            }
            if (sourceIndex % 10 == 0) await Task.Yield();
        }

        if (chunks.Count == 0)
        {
            GatherStatusText = IsEnglish
                ? "Merge failed: no readable training data extracted from enabled sources. Check source type/content (dataset page links are not direct files)."
                : "Merge fallito: nessun dato utile estratto dalle sorgenti abilitate. Controlla tipo/contenuto sorgente (i link pagina dataset non sono file diretti).";
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            GatherIsBusy = false;
            return;
        }

        var mergedRaw = string.Join("\n\n", chunks);
        var mergedDedup = ApplyGatherDedupPolicy(mergedRaw, GatherDedupPolicy);
        await File.WriteAllTextAsync(mergedPath, mergedDedup);
        var outputBytes = Encoding.UTF8.GetByteCount(mergedDedup);
        var meta = new
        {
            createdAtUtc = DateTimeOffset.UtcNow,
            dedupPolicy = GatherDedupPolicy,
            schemaHarmonization = "enabled",
            enabledSources = enabledSources.Count,
            totalSources = GatherSourceEntries.Count,
            outputPath = mergedPath,
            outputCharacters = mergedDedup.Length,
            outputBytes,
            totalFilesScanned,
            totalRecordsRead,
            sources = provenance,
            sourcesWithNoReadableData
        };
        await File.WriteAllTextAsync(mergeMetaPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
        var passReport = new
        {
            generatedAtUtc = DateTimeOffset.UtcNow,
            status = "passed",
            reason = "strict-license-matrix-passed",
            enabledSources = normalizedSources,
            pairChecks = matrixChecks
        };
        await File.WriteAllTextAsync(complianceReportPath, JsonSerializer.Serialize(passReport, new JsonSerializerOptions { WriteIndented = true }));
        GatherProgressValue = 100;
        GatherStagedDatasetPath = mergedPath;
        GatherStatusText = $"Merge completed: {mergedPath} | files={totalFilesScanned:N0} | records={totalRecordsRead:N0} | chars={mergedDedup.Length:N0} | bytes={outputBytes:N0} | elapsed={sw.Elapsed.TotalSeconds:F2}s";
        GatherProgressText = "Merge completed.";
        GatherIsBusy = false;
    }

    private async Task GatherValidateDatasetAsync()
    {
        if (GatherIsBusy) return;
        var sw = Stopwatch.StartNew();
        GatherIsBusy = true;
        GatherProgressValue = 5;
        GatherProgressText = "Validating dataset...";
        var staged = GatherStagedDatasetPath;
        if (string.IsNullOrWhiteSpace(staged))
        {
            GatherValidationText = "No staged dataset.";
            GatherIsBusy = false;
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return;
        }

        var texts = new List<string>();
        if (File.Exists(staged))
        {
            texts.Add(await File.ReadAllTextAsync(staged));
        }
        else if (Directory.Exists(staged))
        {
            var candidates = Directory.EnumerateFiles(staged, "*.*", SearchOption.AllDirectories)
                         .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                                  || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                                  || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
                                  || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
                                  || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase)).ToList();
            var total = Math.Max(1, candidates.Count);
            var idx = 0;
            foreach (var f in candidates)
            {
                texts.Add(await File.ReadAllTextAsync(f));
                idx++;
                GatherProgressValue = 5 + (idx / (double)total) * 80.0;
                if (idx % 20 == 0) await Task.Yield();
            }
        }

        var computed = await Task.Run(() =>
        {
            var raw = string.Join("\n\n", texts);
            var stats = TextCleaner.Analyze(raw);
            var lines = raw.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var uniqueLines = lines.Distinct().Count();
            var dupRatio = lines.Length == 0 ? 0.0 : 1.0 - (uniqueLines / (double)lines.Length);
            var nonAsciiChars = raw.Count(c => c > 127);
            var nonAsciiRatio = raw.Length == 0 ? 0.0 : nonAsciiChars / (double)raw.Length;
            var avgLineLength = lines.Length == 0 ? 0.0 : lines.Average(x => x.Length);
            var metadataOnly = LooksLikeMetadataOnlyDataset(raw);
            var quality = stats.CharacterCount < 5000
                ? "BLOCK"
                : metadataOnly
                    ? "BLOCK"
                : dupRatio > 0.4
                    ? "WARN"
                    : "PASS";
            return (raw, stats, dupRatio, nonAsciiRatio, avgLineLength, quality, metadataOnly);
        });

        var raw = computed.raw;
        var stats = computed.stats;
        var dupRatio = computed.dupRatio;
        var nonAsciiRatio = computed.nonAsciiRatio;
        var avgLineLength = computed.avgLineLength;
        var quality = computed.quality;
        var metadataOnly = computed.metadataOnly;

        GatherRecommendedTokenizer = stats.CharacterCount > 2_000_000 ? "ByteLevelBpe" : "SimpleBpe";
        GatherRecommendedTrainingProfile = stats.CharacterCount > 5_000_000 ? "Serious" : "Balanced";
        GatherValidationText = $"Readiness={quality} | chars={stats.CharacterCount:N0} | lines={stats.LineCount:N0} | dup_ratio={dupRatio:P1} | non_ascii={nonAsciiRatio:P1} | avg_line={avgLineLength:N1} | metadata_only={(metadataOnly ? "yes" : "no")} | tokenizer={GatherRecommendedTokenizer} | profile={GatherRecommendedTrainingProfile}";
        var analyticsPath = Path.Combine(ResolveGatherWorkspace(), "dataset_analytics.json");
        var analytics = new
        {
            createdAtUtc = DateTimeOffset.UtcNow,
            stagedPath = staged,
            readiness = quality,
            stats = new
            {
                stats.CharacterCount,
                stats.LineCount,
                duplicateRatio = dupRatio,
                nonAsciiRatio,
                avgLineLength,
                metadataOnly
            },
            recommendations = new
            {
                tokenizer = GatherRecommendedTokenizer,
                trainingProfile = GatherRecommendedTrainingProfile
            }
        };
        await File.WriteAllTextAsync(analyticsPath, JsonSerializer.Serialize(analytics, new JsonSerializerOptions { WriteIndented = true }));
        GatherStatusText = $"Dataset validation completed | chars={stats.CharacterCount:N0} | lines={stats.LineCount:N0} | elapsed={sw.Elapsed.TotalSeconds:F2}s";
        GatherProgressValue = 100;
        GatherProgressText = "Validation completed.";
        GatherIsBusy = false;
    }
}
