import { ASR_LANGUAGE_OPTIONS, TRANSLATION_TARGET_LANGUAGE_OPTIONS } from "./languages.js";
import { replaceSelectOptions, type SelectOptionSpec } from "./select-options.js";

export class LanguageControls {
  constructor(
    private readonly languageSelect: HTMLSelectElement,
    private readonly translationTargetSelect: HTMLSelectElement,
  ) {}

  get asrLanguage(): string | null {
    return this.languageSelect.value.trim() || null;
  }

  get targetLanguage(): string {
    return this.translationTargetSelect.value.trim();
  }

  get translationEnabled(): boolean {
    return this.targetLanguage !== "";
  }

  render(): void {
    const languageOptions = ASR_LANGUAGE_OPTIONS.map(languageOption);
    const translationTargetOptions = TRANSLATION_TARGET_LANGUAGE_OPTIONS.map(languageOption);
    replaceSelectOptions(
      this.languageSelect,
      [{ value: "", label: "Auto" }, ...languageOptions],
      "",
    );
    replaceSelectOptions(
      this.translationTargetSelect,
      [{ value: "", label: "Off" }, ...translationTargetOptions],
      "",
    );
  }
}

function languageOption(language: string): SelectOptionSpec {
  return { value: language, label: language };
}
