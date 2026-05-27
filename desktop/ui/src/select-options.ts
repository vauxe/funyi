export interface SelectOptionSpec {
  disabled?: boolean;
  label: string;
  title?: string;
  value: string;
}

export function replaceSelectOptions(
  select: HTMLSelectElement,
  options: SelectOptionSpec[],
  selectedValue: string,
): void {
  select.replaceChildren(...options.map(createSelectOption));
  select.value = selectedValue;
}

function createSelectOption({ disabled = false, label, title = "", value }: SelectOptionSpec): HTMLOptionElement {
  const option = document.createElement("option");
  option.disabled = disabled;
  option.textContent = label;
  option.title = title;
  option.value = value;
  return option;
}
