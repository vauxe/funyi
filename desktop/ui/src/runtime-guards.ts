export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}

export function requiredRecord(value: unknown, fieldName: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${fieldName} must be an object`);
  }
  return value;
}

export function optionalRecord(value: unknown, fieldName: string): Record<string, unknown> | null {
  if (value === null || value === undefined) {
    return null;
  }
  return requiredRecord(value, fieldName);
}

export function recordArray(value: unknown, fieldName: string): Record<string, unknown>[] {
  if (value === null || value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} must be an array`);
  }
  return value.map((item) => requiredRecord(item, `${fieldName} item`));
}

export function requiredString(value: unknown, fieldName: string): string {
  if (typeof value !== "string") {
    throw new Error(`${fieldName} must be a string`);
  }
  return value;
}

export function requiredBoolean(value: unknown, fieldName: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${fieldName} must be a boolean`);
  }
  return value;
}
