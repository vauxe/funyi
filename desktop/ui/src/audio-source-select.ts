import { audioSourceDefaultName, audioSourceShortLabel, type AudioSourceKind } from "./audio-source-kind.js";
import type { AudioSource } from "./audio-source.js";
import { replaceSelectOptions, type SelectOptionSpec } from "./select-options.js";

export type SelectableAudioSourceKind = AudioSourceKind | "file";
export interface SelectableAudioSource extends Omit<AudioSource, "kind"> {
  kind: SelectableAudioSourceKind;
}

export class AudioSourceSelect {
  private sourceKinds = new Map<string, SelectableAudioSourceKind>();
  private fallbackUnavailableDetail = "";

  constructor(private readonly select: HTMLSelectElement) {}

  get hasAvailableSource(): boolean {
    return this.sourceKinds.size > 0;
  }

  get unavailableDetail(): string {
    return this.fallbackUnavailableDetail;
  }

  get selectedKind(): SelectableAudioSourceKind | null {
    return this.sourceKinds.get(this.select.value) ?? null;
  }

  render(sources: SelectableAudioSource[]): void {
    this.sourceKinds = new Map(
      sources.filter((source) => source.isAvailable).map((source) => [source.id, source.kind]),
    );
    this.fallbackUnavailableDetail = sources[0]?.detail || "";
    replaceSelectOptions(
      this.select,
      sources.map(audioSourceOption),
      sources.find((source) => source.isAvailable)?.id || "",
    );
  }
}

function audioSourceOption(source: SelectableAudioSource): SelectOptionSpec {
  return {
    disabled: !source.isAvailable,
    label: source.isAvailable ? audioSourceLabel(source) : `${audioSourceLabel(source)} unavailable`,
    title: source.detail || "",
    value: source.id,
  };
}

function audioSourceLabel(source: Pick<SelectableAudioSource, "kind" | "name">): string {
  if (source.kind === "file") {
    return `File · ${source.name.trim() || "Audio file"}`;
  }
  const name = source.name.trim() || audioSourceDefaultName(source.kind);
  return `${audioSourceShortLabel(source.kind)} · ${name}`;
}
