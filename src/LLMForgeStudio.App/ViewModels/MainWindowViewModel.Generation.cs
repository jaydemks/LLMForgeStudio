using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Security.Cryptography;
using Avalonia.Threading;
using LLMForgeStudio.App.Core.Backend;
using LLMForgeStudio.App.Core.Generation;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private void GeneratePreview()
    {
        var fakeLogits = Enumerable.Range(0, Math.Max(1, ModelConfig.VocabSize)).Select(i => Math.Sin(i * 0.37) * 3.0).ToArray();
        var chosen = Sampler.Sample(fakeLogits, SamplingConfig);
        Log = $"Generation preview. {GenerationExplanation}\nChosen token id from fake logits: {chosen}";
    }

    private async Task GenerateFromCheckpointAsync()
    {
        if (IsGenerating) return;
        if (!PythonBackendBridge.IsPythonAvailable(PythonPath))
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "generation.blocked.python",
                "Generation blocked because configured Python executable is unavailable.",
                new { pythonPath = PythonPath, checkpointPath = CheckpointPath, prompt = GenerationPrompt });
            Log = $"Python non trovato: {PythonPath}";
            return;
        }

        IsGenerating = true;
        GeneratedText = string.Empty;
        GenerationStatusText = "Thinking...";

        var resolvedCheckpointPath = ResolveGenerationCheckpointPath();
        if (string.IsNullOrWhiteSpace(resolvedCheckpointPath))
        {
            Log = IsEnglish
                ? "Generation blocked: no valid checkpoint found. Set checkpoint_manifest.json or model.pt."
                : "Generazione bloccata: nessun checkpoint valido trovato. Imposta checkpoint_manifest.json o model.pt.";
            GenerationStatusText = IsEnglish ? "Generation failed: invalid checkpoint." : "Generazione fallita: checkpoint non valido.";
            IsGenerating = false;
            return;
        }

        var projectRoot = ResolveProjectRoot();
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "generate_stub.py");
        var effectiveSeed = SamplingConfig.Seed < 0
            ? RandomNumberGenerator.GetInt32(1, int.MaxValue)
            : SamplingConfig.Seed;
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "generation.start.requested",
            "Generation from checkpoint requested from UI.",
            new
            {
                runDirectory = RunDirectory,
                checkpointPath = CheckpointPath,
                resolvedCheckpointPath = resolvedCheckpointPath,
                prompt = GenerationPrompt,
                effectiveSeed,
                sampling = Clone(SamplingConfig),
                snapshot = BuildProjectPayload()
            });
        var startInfo = PythonBackendBridge.CreateStartInfo(
            PythonPath,
            scriptPath,
            $"--checkpoint \"{resolvedCheckpointPath}\" --prompt \"{GenerationPrompt.Replace("\"", "\\\"")}\" --temperature {SamplingConfig.Temperature.ToString(CultureInfo.InvariantCulture)} --top-k {SamplingConfig.TopK} --seed {effectiveSeed} --max-new-tokens {SamplingConfig.MaxNewTokens}");

        try
        {
            using var process = Process.Start(startInfo);
            if (process is null)
            {
                await _uiDebugLogger.WriteAsync(
                    ResolveRunDirectoryForDebug(),
                    "generation.start.failed",
                    "Generation backend process failed to start.",
                    new { startInfo.FileName, startInfo.Arguments, runDirectory = RunDirectory });
                Log = "Errore avvio generation backend.";
                GenerationStatusText = "Generation failed: backend process could not start.";
                AddNotification("error", IsEnglish ? "Generation failed" : "Generazione fallita", GenerationStatusText);
                return;
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            var err = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                await _uiDebugLogger.WriteAsync(
                    ResolveRunDirectoryForDebug(),
                    "generation.completed.failed",
                    "Generation failed.",
                    new { exitCode = process.ExitCode, stderr = err, checkpointPath = CheckpointPath, prompt = GenerationPrompt });
                Log = $"Generation fallita: {err}";
                GenerationStatusText = "Generation failed.";
                AddNotification("error", IsEnglish ? "Generation failed" : "Generazione fallita", GenerationStatusText);
                return;
            }

            using var doc = JsonDocument.Parse(output);
            var text = doc.RootElement.GetProperty("text").GetString() ?? string.Empty;
            await StreamGeneratedTextAsync(text);
            GenerationStatusText = $"Done. Generated with max new tokens = {SamplingConfig.MaxNewTokens}.";
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "generation.completed.success",
                "Generation completed successfully.",
                new
                {
                    checkpointPath = CheckpointPath,
                    prompt = GenerationPrompt,
                    outputPreview = text.Length > 500 ? text[..500] : text
                });
            Log = "Generation completed successfully.";
            AddNotification("success", IsEnglish ? "Generation completed" : "Generazione completata", GenerationStatusText);
        }
        finally
        {
            IsGenerating = false;
        }
    }

    private string ResolveGenerationCheckpointPath()
    {
        var raw = (CheckpointPath ?? string.Empty).Trim();
        if (!string.IsNullOrWhiteSpace(raw))
        {
            if (File.Exists(raw) && raw.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                return raw;

            if (File.Exists(raw) && raw.EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
            {
                var siblingManifest = Path.Combine(Path.GetDirectoryName(raw) ?? string.Empty, "checkpoint_manifest.json");
                if (File.Exists(siblingManifest))
                    return siblingManifest;
                return raw;
            }
        }

        var runManifest = Path.Combine(RunDirectory, "checkpoint_manifest.json");
        if (File.Exists(runManifest))
            return runManifest;

        var runModel = Path.Combine(RunDirectory, "model.pt");
        if (File.Exists(runModel))
            return runModel;

        return string.Empty;
    }

    private async Task StreamGeneratedTextAsync(string fullText)
    {
        var parts = fullText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
        {
            GeneratedText = string.Empty;
            return;
        }

        var sb = new StringBuilder();
        foreach (var part in parts)
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(part);
            var current = sb.ToString();
            await Dispatcher.UIThread.InvokeAsync(() => { GeneratedText = current; });
            await Task.Delay(18);
        }
    }
}
