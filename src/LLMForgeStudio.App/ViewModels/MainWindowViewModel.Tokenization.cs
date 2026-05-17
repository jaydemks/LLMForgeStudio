using System.Diagnostics;
using System.Text.Json;
using Avalonia.Threading;
using LLMForgeStudio.App.Core.Tokenization;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private static string TruncateForUi(string text, int limit, string marker)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= limit) return text;
        return text[..limit] + "\n\n" + marker;
    }

    private static async Task<T> AwaitOrCancelAsync<T>(Task<T> task, CancellationToken cancellationToken)
    {
        while (true)
        {
            if (task.IsCompleted) return await task;
            var tick = Task.Delay(80, cancellationToken);
            var completed = await Task.WhenAny(task, tick);
            if (completed == task) return await task;
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private async Task SafeAwaitEstimatorAsync(Task task)
    {
        try
        {
            await task;
        }
        catch (OperationCanceledException)
        {
            // expected
        }
    }

    private async Task RunTokenizerProgressEstimatorAsync(
        CancellationToken cancellationToken,
        Stopwatch phaseStopwatch,
        double estimatedSeconds,
        double startProgress,
        double endProgress,
        string phaseText,
        int totalUnits,
        string unitsLabel)
    {
        var safeEstimate = Math.Max(0.5, estimatedSeconds);
        var enteredFinalizing = false;
        var finalizingStartRate = 0.0;
        var finalizingStartProgress = startProgress;
        var smoothedRate = 0.0;
        var finalizingStopwatch = new Stopwatch();
        while (!cancellationToken.IsCancellationRequested)
        {
            var ratio = Math.Clamp(phaseStopwatch.Elapsed.TotalSeconds / safeEstimate, 0, 0.99);
            var progress = startProgress + ((endProgress - startProgress) * ratio);
            var etaSeconds = Math.Max(0, safeEstimate - phaseStopwatch.Elapsed.TotalSeconds);
            var processedUnits = (int)Math.Clamp(Math.Round(totalUnits * ratio), 0, Math.Max(0, totalUnits));
            var rate = processedUnits / Math.Max(0.001, phaseStopwatch.Elapsed.TotalSeconds);
            smoothedRate = smoothedRate <= 0.0 ? rate : ((smoothedRate * 0.8) + (rate * 0.2));
            var isFinalizing = etaSeconds <= 0.9;
            if (isFinalizing && !enteredFinalizing)
            {
                enteredFinalizing = true;
                finalizingStartRate = Math.Max(1.0, smoothedRate);
                finalizingStartProgress = progress;
                finalizingStopwatch.Restart();
            }

            string progressText;
            if (enteredFinalizing)
            {
                var decayProgress = 1.0 - Math.Clamp(smoothedRate / Math.Max(1.0, finalizingStartRate), 0.0, 1.0);
                var easedDecay = 1.0 - Math.Pow(1.0 - decayProgress, 1.8);
                var timeGuard = 1.0 - Math.Exp(-finalizingStopwatch.Elapsed.TotalSeconds / 8.0);
                var blend = Math.Clamp((easedDecay * 0.75) + (timeGuard * 0.25), 0.0, 0.995);
                var finalizingProgress = finalizingStartProgress + ((endProgress - finalizingStartProgress) * blend);
                progress = Math.Max(progress, finalizingProgress);
                var finalizingPct = Math.Clamp(blend * 100.0, 0.0, 99.5);
                progressText = IsEnglish
                    ? $"{phaseText} Finalizing... ~{finalizingPct:0}% of final stage"
                    : $"{phaseText} Finalizzazione... ~{finalizingPct:0}% della fase finale";
            }
            else
            {
                progressText = BuildTokenizerEtaText(phaseText, etaSeconds, true);
            }

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                TokenizerProgressValue = progress;
                TokenizerProgressText = progressText;
                TokenizerLiveStatsText = BuildTokenizerRuntimeStatsText(processedUnits, totalUnits, rate, unitsLabel, phaseStopwatch.Elapsed);
                AppendTokenizerRuntimeLogSnapshot(progress, etaSeconds, processedUnits, totalUnits, rate, unitsLabel);
            });
            await Task.Delay(250, cancellationToken);
        }
    }

    private string BuildTokenizerEtaText(string phaseText, double etaSeconds, bool finalizingWhenNearZero = false)
    {
        if (finalizingWhenNearZero && etaSeconds <= 0.9)
        {
            return IsEnglish
                ? $"{phaseText} Finalizing... This phase may take longer and remaining time is no longer reliable on large datasets."
                : $"{phaseText} Finalizzazione... Questa fase può richiedere più tempo e il tempo residuo non è più stimabile in modo affidabile su dataset grandi.";
        }

        var eta = TimeSpan.FromSeconds(Math.Max(0, etaSeconds));
        var etaText = eta.TotalHours >= 1 ? eta.ToString(@"h\:mm\:ss") : eta.ToString(@"m\:ss");
        return $"{phaseText} ETA ~ {etaText}";
    }

    private string BuildTokenizerRuntimeStatsText(int processedUnits, int totalUnits, double ratePerSec, string unitsLabel, TimeSpan elapsed)
    {
        var elapsedText = elapsed.TotalHours >= 1 ? elapsed.ToString(@"h\:mm\:ss") : elapsed.ToString(@"m\:ss");
        return IsEnglish
            ? $"Processed: {processedUnits:N0}/{totalUnits:N0} {unitsLabel} | Rate: {ratePerSec:N0} {unitsLabel}/s | Elapsed: {elapsedText}"
            : $"Elaborati: {processedUnits:N0}/{totalUnits:N0} {unitsLabel} | Velocità: {ratePerSec:N0} {unitsLabel}/s | Tempo: {elapsedText}";
    }

    private void AppendTokenizerRuntimeLogSnapshot(double progress, double etaSeconds, int processedUnits, int totalUnits, double ratePerSec, string unitsLabel)
    {
        var now = DateTimeOffset.UtcNow;
        if ((now - _lastTokenizerRuntimeLogAtUtc).TotalSeconds < 2.0) return;
        _lastTokenizerRuntimeLogAtUtc = now;
        var eta = TimeSpan.FromSeconds(Math.Max(0, etaSeconds));
        var etaText = eta.TotalHours >= 1 ? eta.ToString(@"h\:mm\:ss") : eta.ToString(@"m\:ss");
        var line = IsEnglish
            ? $"[Tokenizer] {progress:0.0}% | ETA {etaText} | {processedUnits:N0}/{totalUnits:N0} {unitsLabel} | {ratePerSec:N0} {unitsLabel}/s"
            : $"[Tokenizer] {progress:0.0}% | ETA {etaText} | {processedUnits:N0}/{totalUnits:N0} {unitsLabel} | {ratePerSec:N0} {unitsLabel}/s";
        Log = AppendLineWithCap(Log, line, 40_000);
    }

    private static string AppendLineWithCap(string current, string line, int maxChars)
    {
        var baseText = string.IsNullOrWhiteSpace(current) ? string.Empty : current.TrimEnd();
        var next = string.IsNullOrEmpty(baseText) ? line : (baseText + Environment.NewLine + line);
        if (next.Length <= maxChars) return next;
        return next[^maxChars..];
    }

    private double EstimateTokenizerSeconds(int datasetChars)
    {
        var throughput = GetTokenizerThroughput(SelectedTokenizerKind);
        var mergeFactor = 1.0 + (TokenizerConfig.MaxMerges / 800.0);
        var freqFactor = TokenizerConfig.MinFrequency <= 1 ? 1.10 : 1.0;
        var vocabFactor = 1.0 + (TokenizerConfig.TargetVocabSize / 50_000.0);
        var complexity = Math.Max(0.9, mergeFactor * freqFactor * vocabFactor);
        return (Math.Max(1, datasetChars) / Math.Max(10.0, throughput)) * complexity;
    }

    private static double EstimatePreviewSeconds(int tokenCount, int blockSize)
    {
        var workUnits = Math.Max(1, tokenCount) * Math.Max(8, blockSize);
        return Math.Clamp(workUnits / 40_000_000.0, 0.6, 18.0);
    }

    private double GetTokenizerThroughput(TokenizerKind kind)
    {
        if (_tokenizerThroughputCharsPerSec.TryGetValue(kind, out var measured) && measured > 10)
            return measured;

        return kind switch
        {
            TokenizerKind.Character => 2_200_000,
            TokenizerKind.Word => 1_300_000,
            TokenizerKind.ByteLevelBpe => 260_000,
            TokenizerKind.Unigram => 340_000,
            TokenizerKind.WordPiece => 320_000,
            TokenizerKind.SimpleBpe => 280_000,
            TokenizerKind.HybridFallback => 240_000,
            TokenizerKind.HierarchicalExperimental => 500_000,
            _ => 300_000
        };
    }

    private void UpdateTokenizerThroughput(TokenizerKind kind, int chars, double elapsedSeconds)
    {
        if (chars < 1000 || elapsedSeconds <= 0.05) return;
        var measured = chars / elapsedSeconds;
        if (measured <= 10) return;
        if (_tokenizerThroughputCharsPerSec.TryGetValue(kind, out var oldValue))
            _tokenizerThroughputCharsPerSec[kind] = (oldValue * 0.7) + (measured * 0.3);
        else
            _tokenizerThroughputCharsPerSec[kind] = measured;
    }

    private void CancelTokenizerRun()
    {
        if (!IsTokenizerBusy) return;
        _tokenizerCts?.Cancel();
    }

    private sealed class TokenizationState
    {
        public List<int> TokenIds { get; set; } = new();
        public List<VocabularyItem> Vocabulary { get; set; } = new();
        public string DecodedPreview { get; set; } = string.Empty;
    }

    private string ResolveTokenizationStatePath()
    {
        Directory.CreateDirectory(RunDirectory);
        return Path.Combine(RunDirectory, "tokenization_state.json");
    }

    private async Task SaveTokenizationStateAsync()
    {
        if (_lastTokenization is null) return;
        var state = new TokenizationState
        {
            TokenIds = _lastTokenization.TokenIds.ToList(),
            Vocabulary = _lastTokenization.Vocabulary.ToList(),
            DecodedPreview = _lastTokenization.DecodedPreview
        };
        var path = ResolveTokenizationStatePath();
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(state));
    }

    private async Task TryRestoreTokenizationStateAsync()
    {
        var path = ResolveTokenizationStatePath();
        if (!File.Exists(path)) return;
        try
        {
            var raw = await File.ReadAllTextAsync(path);
            var state = JsonSerializer.Deserialize<TokenizationState>(raw);
            if (state is null || state.TokenIds.Count == 0 || state.Vocabulary.Count == 0) return;
            _lastTokenization = new TokenizationResult
            {
                TokenIds = state.TokenIds,
                Vocabulary = state.Vocabulary,
                DecodedPreview = state.DecodedPreview ?? string.Empty
            };
            VocabularyPreview.Clear();
            foreach (var item in _lastTokenization.Vocabulary.Take(200))
                VocabularyPreview.Add(item);
            TokenizerProgressText = IsEnglish ? "Tokenizer restored from saved project state." : "Tokenizer ripristinato dallo stato progetto salvato.";
            TokenizerLiveStatsText = IsEnglish
                ? $"Restored | tokens: {_lastTokenization.TokenCount:N0} | vocab: {_lastTokenization.VocabSize:N0}"
                : $"Ripristinato | token: {_lastTokenization.TokenCount:N0} | vocab: {_lastTokenization.VocabSize:N0}";
            if (string.IsNullOrWhiteSpace(TokenizerStatusText) || TokenizerStatusText.Contains("not trained", StringComparison.OrdinalIgnoreCase))
                TokenizerStatusText = IsEnglish
                    ? $"Tokenizer ready (restored), vocab {_lastTokenization.VocabSize:N0}, tokens {_lastTokenization.TokenCount:N0}."
                    : $"Tokenizer pronto (ripristinato), vocab {_lastTokenization.VocabSize:N0}, token {_lastTokenization.TokenCount:N0}.";
            OnPropertyChanged(nameof(IsTokenizerReady));
        }
        catch
        {
            // best-effort restore
        }
    }

    private static bool LooksLikeMetadataOnlyDataset(string datasetPayload)
    {
        if (string.IsNullOrWhiteSpace(datasetPayload)) return true;
        var sample = datasetPayload.Length > 120_000 ? datasetPayload[..120_000] : datasetPayload;
        var lowered = sample.ToLowerInvariant();
        var metadataSignals = 0;
        if (lowered.Contains("\"_data_files\"")) metadataSignals++;
        if (lowered.Contains("\"_fingerprint\"")) metadataSignals++;
        if (lowered.Contains("\"features\"")) metadataSignals++;
        if (lowered.Contains("task_categories:")) metadataSignals++;
        if (lowered.Contains("size_categories:")) metadataSignals++;
        if (lowered.Contains("dataset statistics")) metadataSignals++;
        if (lowered.Contains("license:")) metadataSignals++;
        if (lowered.Contains("## overview")) metadataSignals++;

        var conversationSignals = 0;
        if (lowered.Contains("\"messages\"")) conversationSignals++;
        if (lowered.Contains("\"prompt\"")) conversationSignals++;
        if (lowered.Contains("\"response\"")) conversationSignals++;
        if (lowered.Contains("\"text\"")) conversationSignals++;
        if (lowered.Contains("user:")) conversationSignals++;
        if (lowered.Contains("assistant:")) conversationSignals++;

        return metadataSignals >= 3 && conversationSignals <= 1;
    }
}
