namespace LLMForgeStudio.App.Core.Training;

public static class CompatibilityRules
{
    public static IReadOnlyList<string> Validate(ModelConfig model, TrainingConfig training, int datasetTokenCount)
    {
        var warnings = new List<string>();

        if (model.Heads <= 0 || model.EmbeddingSize % model.Heads != 0)
            warnings.Add("n_embd deve essere divisibile per n_head. Altrimenti le attention heads non si dividono bene.");

        if (datasetTokenCount > 0 && datasetTokenCount < model.BlockSize * 20)
            warnings.Add("Dataset molto piccolo rispetto al block_size: alto rischio di overfitting/memorizzazione.");

        if (model.VocabSize > 10_000 && model.EmbeddingSize < 256)
            warnings.Add("Vocab size alto con embedding basso: possibile collo di bottiglia informativo.");

        if (model.BlockSize < 64 && model.VocabSize > 2_000)
            warnings.Add("Block size molto piccolo per esperimenti subword/BPE: il contesto potrebbe essere insufficiente.");

        if (training.BatchSize * model.BlockSize > 32768)
            warnings.Add("BatchSize * BlockSize alto: possibile consumo RAM/VRAM importante.");

        var approxParams = ParameterEstimator.Estimate(model);
        var approxModelBytes = approxParams * 4.0;
        var approxTrainingBytes = approxModelBytes * 8.0; // params + grads + optimizer states (stima)
        var approxTrainingGb = approxTrainingBytes / (1024d * 1024d * 1024d);
        if (approxTrainingGb > 10)
            warnings.Add($"Stima memoria training alta (~{approxTrainingGb:F1} GB). Verifica VRAM/RAM disponibile.");

        return warnings;
    }
}
