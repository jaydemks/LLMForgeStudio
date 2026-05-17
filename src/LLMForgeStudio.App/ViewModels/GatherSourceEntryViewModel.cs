namespace LLMForgeStudio.App.ViewModels;

public sealed class GatherSourceEntryViewModel : ObservableObject
{
    private bool _isEnabled = true;
    private int _weight = 1;
    private bool _isLicensePermitted;

    public GatherSourceEntryViewModel(string path, string provider, string licenseLabel, bool isLicensePermitted)
    {
        Path = path;
        Provider = provider;
        LicenseLabel = licenseLabel;
        _isLicensePermitted = isLicensePermitted;
    }

    public string Path { get; }
    public string Provider { get; }
    public string LicenseLabel { get; }

    public bool IsEnabled
    {
        get => _isEnabled;
        set => SetProperty(ref _isEnabled, value);
    }

    public int Weight
    {
        get => _weight;
        set => SetProperty(ref _weight, value < 1 ? 1 : value);
    }

    public bool IsLicensePermitted
    {
        get => _isLicensePermitted;
        set => SetProperty(ref _isLicensePermitted, value);
    }
}
