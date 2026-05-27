import { audioSourceDefaultName, audioSourceShortLabel, type AudioSourceKind } from "./audio-source-kind.js";
import type { AudioSource } from "./audio-source.js";
import { replaceSelectOptions, type SelectOptionSpec } from "./select-options.js";

export class AudioSourceSelect {
  private sourceKinds = new Map<string, AudioSourceKind>();
  private fallbackUnavailableDetail = "";

  constructor(private readonly select: HTMLSelectElement) {}

  get hasAvailableSource(): boolean {
    return this.sourceKinds.size > 0;
  }

  get unavailableDetail(): string {
    return this.fallbackUnavailableDetail;
  }

  get selectedKind(): AudioSourceKind | null {
    return this.sourceKinds.get(this.select.value) ?? null;
  }

  render(sources: AudioSource[]): void {
    this.sourceKinds = new Map(
      sources
        .filter((source) => source.isAvailable)
        .map((source) => [source.id, source.kind]),
    );
    this.fallbackUnavailableDetail = sources[0]?.detail || "";
    replaceSelectOptions(
      this.select,
      sources.map(audioSourceOption),
      sources.find((source) => source.isAvailable)?.id || "",
    );
  }
}

function audioSourceOption(source: AudioSource): SelectOptionSpec {
  return {
    disabled: !source.isAvailable,
    label: source.isAvailable ? audioSourceLabel(source) : `${audioSourceLabel(source)} unavailable`,
    title: source.detail || "",
    value: source.id,
  };
}

function audioSourceLabel(source: Pick<AudioSource, "kind" | "name">): string {
  const name = source.name.trim() || audioSourceDefaultName(source.kind);
  return `${audioSourceShortLabel(source.kind)} · ${name}`;
}
