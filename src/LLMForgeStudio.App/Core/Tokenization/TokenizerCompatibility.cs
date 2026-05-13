namespace LLMForgeStudio.App.Core.Tokenization;

public static class TokenizerCompatibility
{
    public static IReadOnlyList<string> Validate(TokenizerConfig tokenizer, int datasetChars, int blockSize)
    {
        var warnings = new List<string>();

        if (tokenizer.Kind == TokenizerKind.Character && blockSize < 256)
            warnings.Add("Character-level con block_size basso vede pochissimo contesto: utile per demo, debole per testo lungo.");

        if (tokenizer.Kind == TokenizerKind.SimpleBpe && tokenizer.TargetVocabSize < 256)
            warnings.Add("BPE con vocab troppo piccolo rischia di comportarsi quasi come character-level.");

        if (tokenizer.Kind == TokenizerKind.SimpleBpe && datasetChars < 50_000)
            warnings.Add("Dataset piccolo per BPE: le merge potrebbero essere poco rappresentative.");

        if (tokenizer.Kind == TokenizerKind.HierarchicalExperimental)
            warnings.Add("Tokenizer gerarchico segnato come esperimento: non usarlo per training serio finché non esiste backend dedicato.");

        return warnings;
    }
}
