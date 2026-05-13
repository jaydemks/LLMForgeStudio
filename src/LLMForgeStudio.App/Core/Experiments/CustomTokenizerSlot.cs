namespace LLMForgeStudio.App.Core.Experiments;

public sealed class CustomTokenizerSlot
{
    public string Name { get; set; } = "Hierarchical Tokenizer Lab";
    public bool Enabled { get; set; } = false;
    public string Concept { get; set; } = "macro-token + subword + char fallback + future metadata vector";
    public string Warning { get; set; } = "Experimental: design idea only, not production training.";
}
