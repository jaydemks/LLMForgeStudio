namespace LLMForgeStudio.App.Core.Training;

public static class TrainingBatchBuilder
{
    public static BatchPreview BuildPreview(IReadOnlyList<int> tokenIds, int blockSize)
    {
        if (tokenIds.Count < 2)
            return new BatchPreview(Array.Empty<int>(), Array.Empty<int>(), "Servono almeno 2 token per creare x/y.");

        var usable = Math.Min(blockSize, tokenIds.Count - 1);
        var x = tokenIds.Take(usable).ToList();
        var y = tokenIds.Skip(1).Take(usable).ToList();

        // Questo è il punto più importante del training loop.
        // x = quello che il modello vede.
        // y = quello che deve indovinare.
        // y è x spostato avanti di 1 token.
        var explanation = "x è la finestra visibile. y è la stessa finestra spostata avanti di 1. Il modello impara next-token prediction.";
        return new BatchPreview(x, y, explanation);
    }
}
