# User Story <epic>.<story>: <Title>

## Story

**As a** <role>
**I want** <capability>
**So that** <outcome>

## Background

<Context for why this story exists. What system, constraint, or prior art does it relate to?
Reference existing tnkr modules (with file paths), upstream projects (lerobot, etc.), or
sibling stories this builds on (e.g. "Builds on 1.1; assumes ports have been discovered").>

## Acceptance Criteria

### Core Functionality

- [ ] **<Capability A>**: <description — what behaviour is observable when this passes>
- [ ] **<Capability B>**: <description>
- [ ] **<Capability C>**: <description>

### User Experience

- [ ] **<UX requirement>**: <description>
- [ ] **<UX requirement>**: <description>

### Technical Requirements

- [ ] **<Tech requirement>**: <description>
- [ ] **<Tech requirement>**: <description>

## Expected User Flow

```bash
$ <command>

<expected output / interaction>
```

<Optional secondary flow, e.g. multi-arm, multi-device, or alternate path:>

```bash
$ <command>
# Follow prompts...
```

## Implementation Details

### File Structure

```
<path/to/feature>/
├── <file>.ts                      # <purpose>
└── <file>.ts                      # <purpose>
```

### Key Dependencies

- **<package>**: <why>
- **<package>**: <why>

### Core Functions to Implement

```typescript
// <file>.ts
async function <name>(): Promise<<type>>;
async function <name>(): Promise<<type>>;
```

### Technical Considerations

#### <Topic, e.g. Cross-Platform Behavior>

- **<Platform / case>**: <detail>
- **<Platform / case>**: <detail>

#### Error Scenarios to Handle

- <scenario>
- <scenario>
- <scenario>

#### Performance & UX

- <constraint, e.g. "Port scan < 1s">
- <constraint>
- <constraint>

## Definition of Done

- [ ] **Functional**: <what passing looks like end-to-end>
- [ ] **Tested**: <unit / integration / e2e expectations>
- [ ] **Documented**: <README / docs section updated>
- [ ] **CLI Ready**: <if applicable — installable and runnable>
- [ ] **Type Safe**: Full TypeScript coverage with strict mode
- [ ] **Follows Conventions**: <naming, file layout, lint/type-check clean>
