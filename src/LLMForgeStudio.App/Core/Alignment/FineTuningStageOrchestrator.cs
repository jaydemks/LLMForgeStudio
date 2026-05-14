using System.Text.Json;
using System.Text;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.Core.Alignment;

public static class FineTuningStageOrchestrator
{
    public static async Task<string> PrepareAsync(string rawDataset, TrainingConfig training, string runDirectory)
    {
        Directory.CreateDirectory(runDirectory);

        var current = rawDataset ?? string.Empty;
        var completed = new List<string>();

        if (training.FineTuningOrchestration)
        {
            if (training.FineTuneStageSft)
            {
                current = SftDatasetFormatter.FormatToTrainingText(current);
                completed.Add("sft");
            }

            if (training.FineTuneStageDpo)
            {
                current = DpoDatasetFormatter.FormatToTrainingText(current);
                completed.Add("dpo");
            }

            if (training.FineTuneStageRlhf)
            {
                var importedCount = 0;
                var imported = TryLoadRlhfFeedback(training.RlhfFeedbackPath, out importedCount);
                current = ApplyRlhfPlaceholder(current, training.RlhfFeedbackSource, imported);
                completed.Add("rlhf");
                var importMetaPath = Path.Combine(runDirectory, "rlhf_feedback_import.json");
                var importMeta = new
                {
                    source = training.RlhfFeedbackSource,
                    path = training.RlhfFeedbackPath,
                    imported_records = importedCount
                };
                await File.WriteAllTextAsync(importMetaPath, JsonSerializer.Serialize(importMeta, new JsonSerializerOptions { WriteIndented = true }));
            }
        }
        else
        {
            current = training.AlignmentMode.ToLowerInvariant() switch
            {
                "sft" => SftDatasetFormatter.FormatToTrainingText(current),
                "dpo" => DpoDatasetFormatter.FormatToTrainingText(current),
                _ => current
            };
            completed.Add(training.AlignmentMode.ToLowerInvariant());
        }

        var outPath = Path.Combine(runDirectory, "dataset_consolidated.txt");
        await File.WriteAllTextAsync(outPath, current);

        var stagePath = Path.Combine(runDirectory, "fine_tuning_stages.json");
        var payload = new
        {
            enabled = training.FineTuningOrchestration,
            alignment_mode = training.AlignmentMode,
            completed_stages = completed.Where(x => !string.IsNullOrWhiteSpace(x)).ToArray(),
            rlhf_feedback_source = training.RlhfFeedbackSource,
            reward_modeling_enabled = training.RewardModelingEnabled,
            safety_policy_mode = training.SafetyPolicyMode
        };
        await File.WriteAllTextAsync(stagePath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));

        return outPath;
    }

    private static string ApplyRlhfPlaceholder(string text, string feedbackSource, string importedFeedback)
    {
        var baseText = text ?? string.Empty;
        var sb = new StringBuilder();
        sb.Append("<|rlhf_feedback_source|>").Append(feedbackSource).Append('\n');
        if (!string.IsNullOrWhiteSpace(importedFeedback))
            sb.Append(importedFeedback.Trim()).Append('\n');
        sb.Append(baseText);
        return sb.ToString();
    }

    private static string TryLoadRlhfFeedback(string feedbackPath, out int importedCount)
    {
        importedCount = 0;
        if (string.IsNullOrWhiteSpace(feedbackPath) || !File.Exists(feedbackPath))
            return string.Empty;

        var lines = File.ReadAllLines(feedbackPath);
        var sb = new StringBuilder();
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0) continue;
            try
            {
                using var doc = JsonDocument.Parse(trimmed);
                var root = doc.RootElement;
                var prompt = root.TryGetProperty("prompt", out var p) ? p.GetString() ?? string.Empty : string.Empty;
                var chosen = root.TryGetProperty("chosen", out var c) ? c.GetString() ?? string.Empty : string.Empty;
                var rejected = root.TryGetProperty("rejected", out var r) ? r.GetString() ?? string.Empty : string.Empty;
                if (string.IsNullOrWhiteSpace(prompt) || string.IsNullOrWhiteSpace(chosen)) continue;

                if (sb.Length > 0) sb.AppendLine();
                sb.Append("<|rlhf_user|>\n").Append(prompt).Append('\n');
                sb.Append("<|rlhf_preferred|>\n").Append(chosen).Append('\n');
                if (!string.IsNullOrWhiteSpace(rejected))
                    sb.Append("<|rlhf_rejected|>\n").Append(rejected).Append('\n');
                importedCount++;
            }
            catch
            {
                // ignore invalid feedback lines
            }
        }

        return sb.ToString();
    }
}
