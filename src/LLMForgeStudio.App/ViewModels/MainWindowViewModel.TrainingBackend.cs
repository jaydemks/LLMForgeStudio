using System.Diagnostics;
using System.Text.Json;
using Avalonia.Threading;
using LLMForgeStudio.App.Core.Backend;
using LLMForgeStudio.App.Core.Cluster;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private static readonly TimeSpan TrainingNoProgressNoticeAfter = TimeSpan.FromSeconds(20);
    private static readonly TimeSpan TrainingNoProgressHardTimeout = TimeSpan.FromMinutes(15);

    private void RunDryTraining()
    {
        if (_lastTokenization is null) _ = TrainTokenizerAsync();
        TrainingLogs.Clear();
        foreach (var entry in DryRunTrainer.Simulate(TrainingConfig, _lastTokenization?.TokenCount ?? 0))
            TrainingLogs.Add(entry);
        Log = "Dry-run completed.";
        OnPropertyChanged(nameof(ChartSummary));
    }

    private async Task StartBackendTrainingAsync()
    {
        if (IsTraining) return;
        if (IsTokenizerBusy)
        {
            TrainingStatusText = IsEnglish ? "Training blocked: tokenizer is still running." : "Training bloccato: tokenizer ancora in esecuzione.";
            Log = TrainingStatusText;
            AddNotification("warning", IsEnglish ? "Training blocked" : "Training bloccato", TrainingStatusText);
            return;
        }
        NormalizeTrainingRuntimeBeforeLaunch();
        if (!IsTrainingReady)
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.missing_steps",
                "Training blocked because required preparation steps are missing.",
                new { missing = GetTrainingMissingSteps(), snapshot = BuildProjectPayload() });
            Log = $"Training blocked. Missing steps: {string.Join(", ", GetTrainingMissingSteps())}.";
            AddNotification("warning", IsEnglish ? "Training blocked" : "Training bloccato", Log);
            return;
        }
        if (!PythonBackendBridge.IsPythonAvailable(PythonPath))
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.python",
                "Training blocked because configured Python executable is unavailable.",
                new { pythonPath = PythonPath, snapshot = BuildProjectPayload() });
            Log = $"Python non trovato: {PythonPath}";
            AddNotification("error", IsEnglish ? "Python not found" : "Python non trovato", Log);
            return;
        }
        var preflightBlock = GetBlockingPreflightIssue();
        if (!string.IsNullOrWhiteSpace(preflightBlock))
        {
            TrainingStatusText = "Training blocked by preflight checks.";
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.preflight",
                "Training blocked by preflight checks.",
                new { preflightBlock, snapshot = BuildProjectPayload() });
            Log = preflightBlock;
            AddNotification("warning", IsEnglish ? "Preflight blocked training" : "Preflight ha bloccato il training", preflightBlock);
            return;
        }

        var projectRoot = ResolveProjectRoot();
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "cluster_runner.py");
        await PrepareRlhfFeedbackForRunAsync();
        string datasetForBackend;
        try
        {
            datasetForBackend = await WriteConsolidatedDatasetForBackendAsync();
        }
        catch (InvalidOperationException ex)
        {
            TrainingStatusText = "Training blocked: dataset preflight failed.";
            Log = ex.Message;
            AddNotification("warning", IsEnglish ? "Dataset quality gate blocked training" : "Quality gate dataset ha bloccato il training", ex.Message);
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.dataset_preflight",
                "Training blocked before backend launch because dataset resolved empty.",
                new { error = ex.Message, runDirectory = RunDirectory, datasetPath = DatasetPath, gatherStagedDatasetPath = GatherStagedDatasetPath });
            return;
        }
        var spec = new BackendJobSpec
        {
            JobType = "train",
            DatasetPath = datasetForBackend,
            OutputDirectory = RunDirectory,
            Tokenizer = TokenizerConfig,
            Model = ModelConfig,
            Training = TrainingConfig,
            Sampling = SamplingConfig,
            ClusterProfileName = TrainingConfig.ClusterProfileName
        };

        var specPath = await PythonBackendBridge.WriteJobSpecAsync(spec, RunDirectory);
        var clusterDescriptor = ClusterJobDescriptor.FromSpec(spec);
        clusterDescriptor.JobSpecPath = specPath;
        var clusterDescriptorPath = await ClusterJobDescriptor.WriteAsync(RunDirectory, clusterDescriptor);
        spec.ClusterJobDescriptorPath = clusterDescriptorPath;
        specPath = await PythonBackendBridge.WriteJobSpecAsync(spec, RunDirectory);
        var startInfo = PythonBackendBridge.CreateStartInfo(PythonPath, scriptPath, $"--job \"{specPath}\"");
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.start.requested",
            "Training process requested from UI.",
            new
            {
                runDirectory = RunDirectory,
                projectRoot,
                scriptPath,
                specPath,
                clusterDescriptorPath,
                snapshot = BuildProjectPayload()
            });

        _trainingProcess = Process.Start(startInfo);
        if (_trainingProcess is null)
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.start.failed",
                "Backend process failed to start.",
                new { startInfo.FileName, startInfo.Arguments, runDirectory = RunDirectory });
            Log = "Errore avvio processo training.";
            TrainingStatusText = "Training failed: backend process could not start.";
            return;
        }

        var stdOutTask = _trainingProcess.StandardOutput.ReadToEndAsync();
        var stdErrTask = _trainingProcess.StandardError.ReadToEndAsync();

        IsTraining = true;
        _trainingStartedAtUtc = DateTimeOffset.UtcNow;
        _trainingLastProgressAtUtc = _trainingStartedAtUtc.Value;
        _trainingMonitorCts?.Cancel();
        _trainingMonitorCts?.Dispose();
        _trainingMonitorCts = new CancellationTokenSource();
        _wizardTrainingStarted = true;
        TrainingLogs.Clear();
        TrainingStatusText = "Training started. Waiting for backend logs...";
        Log = $"Training started. Output directory: {RunDirectory}\nConfigured runtime: steps={TrainingConfig.MaxSteps}, eval_every={TrainingConfig.EvalEvery}, batch={TrainingConfig.BatchSize}, grad_accum={TrainingConfig.GradientAccumulationSteps}.";
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.started",
            "Training process started.",
            new { processId = _trainingProcess.Id, runDirectory = RunDirectory });
        RaiseTrainingDashboardChanged();
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));

        _ = MonitorBackendTrainingAsync(_trainingProcess, stdOutTask, stdErrTask, _trainingMonitorCts.Token);
    }

    private void NormalizeTrainingRuntimeBeforeLaunch()
    {
        var profile = (SelectedTrainingProfile ?? string.Empty).Trim();
        if (profile.Equals("Serious", StringComparison.OrdinalIgnoreCase) && TrainingConfig.MaxSteps < 1800)
        {
            TrainingConfig.MaxSteps = 1800;
            Log = IsEnglish
                ? "Serious profile safeguard: MaxSteps auto-raised to 1800 before launch."
                : "Protezione profilo Serious: MaxSteps alzato automaticamente a 1800 prima del lancio.";
            OnPropertyChanged(nameof(TrainingMaxSteps));
        }

        if (TrainingConfig.EvalEvery <= 0)
        {
            TrainingConfig.EvalEvery = 100;
            OnPropertyChanged(nameof(TrainingEvalEvery));
        }
    }

    private async Task MonitorBackendTrainingAsync(Process process, Task<string> stdOutTask, Task<string> stdErrTask, CancellationToken cancellationToken)
    {
        try
        {
            var stallWarned = false;
            while (!process.HasExited)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var progressed = await PollTrainingLogAsync();
                if (progressed)
                {
                    _trainingLastProgressAtUtc = DateTimeOffset.UtcNow;
                    stallWarned = false;
                }

                var now = DateTimeOffset.UtcNow;
                var idleFor = now - _trainingLastProgressAtUtc;
                var elapsed = _trainingStartedAtUtc.HasValue ? now - _trainingStartedAtUtc.Value : TimeSpan.Zero;
                var lastObservedStep = TrainingLogs.Count > 0 ? TrainingLogs.Last().Step : -1;
                var targetSteps = Math.Max(1, TrainingConfig.MaxSteps);
                var pct = lastObservedStep >= 0 ? Math.Clamp((lastObservedStep / (double)targetSteps) * 100.0, 0, 100) : 0;
                if (TrainingLogs.Count == 0)
                {
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        TrainingStatusText = $"Training running... waiting for first log entry. Elapsed {elapsed:mm\\:ss}.";
                    });
                }
                else if (idleFor >= TrainingNoProgressNoticeAfter)
                {
                    stallWarned = true;
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        TrainingStatusText = $"Training finalizing... no new log lines for {idleFor:mm\\:ss} (last step {lastObservedStep}/{targetSteps}, {pct:F1}%).";
                    });
                }

                if (idleFor >= TrainingNoProgressHardTimeout)
                {
                    try { process.Kill(entireProcessTree: true); } catch { /* ignored */ }
                    throw new TimeoutException(
                        $"Training watchdog timeout: no backend progress for {idleFor:mm\\:ss} (last step {lastObservedStep}/{targetSteps}). Process was terminated.");
                }

                await Task.Delay(400, cancellationToken);
            }

            await PollTrainingLogAsync();

            var stdOut = await stdOutTask;
            var stdErr = await stdErrTask;
            var exitCode = process.ExitCode;
            var modelPath = Path.Combine(RunDirectory, "model.pt");
            var manifestPath = Path.Combine(RunDirectory, "checkpoint_manifest.json");
            var lastStep = TrainingLogs.Count > 0 ? TrainingLogs.Last().Step : -1;
            var hasArtifacts = File.Exists(modelPath) && File.Exists(manifestPath);
            var likelyCompleted = hasArtifacts && lastStep >= TrainingConfig.MaxSteps;

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (exitCode == 0)
                {
                    _ = RunArtifactRegistry.UpdateAsync(RunDirectory);
                    _ = RefreshEvalSummaryAsync();
                    TrainingStatusText = $"Training completed successfully. Steps logged: {TrainingLogs.Count}.";
                    Log = $"Training completed. Output: {RunDirectory}";
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.success",
                        "Training completed successfully.",
                        new { exitCode, runDirectory = RunDirectory, modelPath, manifestPath, lastStep, hasArtifacts });
                    AutoSelectCheckpointIfAvailable();
                    ComputeTrainingAdvisorFromLogs();
                    AddNotification("success", IsEnglish ? "Training completed" : "Training completato", TrainingStatusText);
                }
                else if (likelyCompleted)
                {
                    _ = RunArtifactRegistry.UpdateAsync(RunDirectory);
                    _ = RefreshEvalSummaryAsync();
                    TrainingStatusText = $"Training completed (with non-zero exit code {exitCode}). Artifacts are valid.";
                    Log = $"Training completed with warning (exit code {exitCode}) but checkpoint artifacts were produced.\nOutput: {RunDirectory}";
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.warning",
                        "Training exited with non-zero code but artifacts look valid.",
                        new { exitCode, runDirectory = RunDirectory, modelPath, manifestPath, lastStep, hasArtifacts });
                    AutoSelectCheckpointIfAvailable();
                    ComputeTrainingAdvisorFromLogs();
                    AddNotification("warning", IsEnglish ? "Training completed with warning" : "Training completato con avviso", TrainingStatusText);
                }
                else
                {
                    var err = string.IsNullOrWhiteSpace(stdErr) ? stdOut : stdErr;
                    var compactErr = string.IsNullOrWhiteSpace(err) ? "(no backend error output)" : err.Trim();
                    var normalizedErr = TryNormalizeBackendError(compactErr);
                    TrainingStatusText = $"Training failed (exit code {exitCode}).";
                    Log = $"Training failed (exit code {exitCode}).\nBackend output:\n{normalizedErr}";
                    ClearAllTrainingAdvisor();
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.failed",
                        "Training failed.",
                        new { exitCode, runDirectory = RunDirectory, backendError = normalizedErr, modelPath, manifestPath, lastStep, hasArtifacts });
                    AddNotification("error", IsEnglish ? "Training failed" : "Training fallito", TrainingStatusText);
                }

                if (stallWarned && exitCode == 0)
                {
                    Log += "\nWatchdog note: backend had a long finalization phase before clean exit.";
                }
            });

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                OnPropertyChanged(nameof(CanExportForOllama));
                OnPropertyChanged(nameof(OllamaExportButtonHint));
            });
        }
        catch (OperationCanceledException)
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                TrainingStatusText = "Training cancelled.";
                if (!Log.Contains("cancelled", StringComparison.OrdinalIgnoreCase))
                    Log = "Training cancelled by user.";
            });
        }
        catch (Exception ex)
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                TrainingStatusText = "Training monitor failed.";
                Log = $"Training monitor error: {ex.Message}";
            });
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.monitor.error",
                "Training monitor threw an exception.",
                new { exception = ex.ToString(), runDirectory = RunDirectory });
        }
        finally
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                IsTraining = false;
                RaiseTrainingDashboardChanged();
            });
            _trainingMonitorCts?.Dispose();
            _trainingMonitorCts = null;
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.monitor.closed",
                "Training monitor closed.",
                new { runDirectory = RunDirectory, logEntries = TrainingLogs.Count });
        }
    }

    private async Task<bool> PollTrainingLogAsync()
    {
        var logPath = Path.Combine(RunDirectory, "train_log.jsonl");
        if (!File.Exists(logPath)) return false;

        string[] lines;
        try
        {
            await using var fs = new FileStream(logPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var sr = new StreamReader(fs);
            var content = await sr.ReadToEndAsync();
            lines = content.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        }
        catch (IOException)
        {
            return false;
        }

        var parsed = new List<TrainingLogEntry>();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var cleanLine = line.Trim().TrimStart('\0', '\uFEFF');
            if (cleanLine.Length == 0) continue;

            try
            {
                using var doc = JsonDocument.Parse(cleanLine);
                var root = doc.RootElement;
                parsed.Add(new TrainingLogEntry(
                    root.GetProperty("step").GetInt32(),
                    root.GetProperty("train_loss").GetDouble(),
                    root.GetProperty("val_loss").GetDouble(),
                    root.GetProperty("tokens_per_second").GetDouble(),
                    root.TryGetProperty("message", out var message) ? message.GetString() ?? string.Empty : string.Empty));
            }
            catch (JsonException)
            {
                // skip malformed line while file is still being written
            }
        }

        var previousCount = TrainingLogs.Count;
        var previousLastStep = previousCount > 0 ? TrainingLogs.Last().Step : -1;
        await Dispatcher.UIThread.InvokeAsync(() =>
        {
            TrainingLogs.Clear();
            foreach (var item in parsed.TakeLast(500)) TrainingLogs.Add(item);
            if (TrainingLogs.Count > 0)
            {
                var last = TrainingLogs.Last();
                TrainingStatusText = $"Training running... step {last.Step}, train loss {last.TrainLoss:F4}, val loss {last.ValLoss:F4}.";
            }
            RaiseTrainingDashboardChanged();
            OnPropertyChanged(nameof(ChartSummary));
        });

        var currentCount = parsed.Count;
        var currentLastStep = currentCount > 0 ? parsed[^1].Step : -1;
        return currentCount != previousCount || currentLastStep != previousLastStep;
    }

    private void CancelBackendTraining()
    {
        if (_trainingProcess is null || _trainingProcess.HasExited) return;
        _trainingMonitorCts?.Cancel();
        _trainingProcess.Kill(entireProcessTree: true);
        _ = WriteCancelledClusterStateArtifactsAsync();
        Log = "Training cancelled by user.";
        TrainingStatusText = "Training cancelled.";
        IsTraining = false;
        _ = _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.cancelled",
            "Training cancelled by user.",
            new { runDirectory = RunDirectory, processId = _trainingProcess.Id });
        RaiseTrainingDashboardChanged();
    }

    private async Task WriteCancelledClusterStateArtifactsAsync()
    {
        try
        {
            var statePath = Path.Combine(RunDirectory, "cluster_run_state.json");
            var heartbeatPath = Path.Combine(RunDirectory, "cluster_heartbeat.json");
            var now = DateTimeOffset.UtcNow;

            var statePayload = new
            {
                status = "cancelled",
                updated_at_utc = now,
                exit_code = -1,
                note = "Cancelled by user from UI"
            };
            var hbPayload = new
            {
                status = "stopped",
                updated_at_utc = now,
                note = "Cancelled by user from UI"
            };

            await File.WriteAllTextAsync(statePath, JsonSerializer.Serialize(statePayload, new JsonSerializerOptions { WriteIndented = true }));
            await File.WriteAllTextAsync(heartbeatPath, JsonSerializer.Serialize(hbPayload, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
            // Best effort only: training cancel should never fail because artifact update failed.
        }
    }

    private static string TryNormalizeBackendError(string backendText)
    {
        if (string.IsNullOrWhiteSpace(backendText)) return backendText;
        const string marker = "LLMFORGE_ERROR: ";
        var line = backendText
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .Select(x => x.Trim())
            .FirstOrDefault(x => x.StartsWith(marker, StringComparison.Ordinal));
        if (line is null) return backendText;

        try
        {
            var payloadJson = line[marker.Length..];
            using var doc = JsonDocument.Parse(payloadJson);
            var code = doc.RootElement.TryGetProperty("error_code", out var c) ? (c.GetString() ?? "RUNTIME_EXCEPTION") : "RUNTIME_EXCEPTION";
            var msg = doc.RootElement.TryGetProperty("message", out var m) ? (m.GetString() ?? "Unknown error") : "Unknown error";
            var hint = code switch
            {
                "DATASET_EMPTY" => "Dataset appears empty after cleaning/loading. Re-import dataset and re-run merge/validate.",
                "TRAIN_WINDOW_INVALID" => "Dataset token window is too short for current configuration. Reduce block size or use a larger dataset.",
                "TOKENIZER_STATE_REQUIRED" => "Selected tokenizer state is missing/invalid. Run Tokenization first, then retry Training.",
                _ => "Check backend logs and training configuration."
            };
            return $"[{code}] {msg}\nHint: {hint}";
        }
        catch
        {
            return backendText;
        }
    }
}
