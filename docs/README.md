# Documentation Organization Guide

This documentation structure is optimized for AI agents to quickly understand the project context and find relevant information.

## Directory Structure

```
docs/
├── README.md                     # This file - documentation guide
├── 01-project-overview/          # High-level project information and goals
├── 02-technical-guides/          # Implementation guides and how-to documentation
├── 03-performance-analysis/      # Performance studies and optimization reports
├── 04-features/                  # Individual feature specifications and designs
├── 05-operations/               # Commands, procedures, and operational guides
└── 99-archive/                  # Deprecated or outdated documentation
```

## Reading Order for AI Agents

For comprehensive understanding of this project, AI agents should read documentation in this order:

1. **Start here**: `/workspace/README.md` (main project README)
2. **Project context**: `01-project-overview/` - Understand goals and scope
3. **Technical foundation**: `02-technical-guides/` - Learn how systems work
4. **Performance context**: `03-performance-analysis/` - Understand constraints and optimizations
5. **Feature details**: `04-features/` - Specific feature implementations
6. **Operations**: `05-operations/` - How to run and manage the system

## Current Contents

### 02-technical-guides/
- `Google_Drive_Integration_Guide.md` - Google Drive integration and automation setup
- `Lesson_Organization_System.md` - Lesson structure and organization system

### 03-performance-analysis/
- `File_Operations_Performance_Analysis.md` - File operation performance analysis
- `System_Performance_Optimizations.md` - System optimization strategies and results

### 04-features/
- `Cross_Lesson_Context_System_Design.md` - Technical plan for cross-lesson context enhancement

### Lesson-Specific Documentation
Lesson READMEs remain in their respective lesson directories:
- `lessons/TLP/README.md`
- `lessons/Conducting_Operations_in_a_Degraded_Space/README.md`
- `lessons/Perform_Effectively_In_An_Operational_Environment/README.md`

## Guidelines for Adding Documentation

- **Project-wide documentation**: Use numbered directories for logical reading order
- **Feature-specific documentation**: Place in `04-features/` with descriptive naming
- **Operational procedures**: Place in `05-operations/`
- **Deprecated content**: Move to `99-archive/` rather than deleting
- **Lesson-specific content**: Keep in individual lesson directories

This organization ensures AI agents can quickly navigate from high-level understanding to specific implementation details.