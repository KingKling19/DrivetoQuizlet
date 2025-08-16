# Documentation Organization Guide

## Overview
This guide explains the reorganized documentation structure for the DriveToQuizlet military training lesson processor project. All markdown files have been systematically organized by purpose to enable easy navigation and understanding.

## Directory Structure

### `/docs/overview/` - Project Overview
Contains high-level project information and core documentation:
- `project-overview.md` - Main product brief, target audience, features, and architecture
- `documentation-organization.md` - This file explaining the organization structure

### `/docs/planning/` - Development Planning
Contains project planning, roadmaps, and evaluation documents:
- `development-plan.md` - Comprehensive development plan with phases and features
- `features-1-7-evaluation.md` - Evaluation summary of features 1-7 implementation

### `/docs/phases/` - Implementation Phases
Contains detailed implementation summaries for each development phase:
- `phase1-foundation-implementation.md` - Phase 1: Foundation & infrastructure setup
- `phase1-optimization-implementation.md` - Phase 1: Performance optimizations
- `phase2-automation-implementation.md` - Phase 2: Core automation features
- `phase2a-drive-integration-implementation.md` - Phase 2A: Google Drive integration
- `phase2b-content-processing-implementation.md` - Phase 2B: Content processing pipeline
- `phase3-ai-integration-implementation.md` - Phase 3: AI integration and analysis
- `phase3-flashcard-review-implementation.md` - Phase 3: Flashcard review system

### `/docs/implementation/` - Technical Implementation
Contains technical documentation about specific implementations:
- `performance-optimizations.md` - Performance improvement strategies and results
- `file-copying-performance.md` - File copying optimization details
- `google-drive-automation.md` - Google Drive automation technical documentation

### `/docs/guides/` - User Guides & Setup
Contains user-facing guides and setup instructions:
- `ui-setup-guide.md` - User interface setup and configuration guide
- `lesson-organization.md` - Guide for organizing lesson content and materials

### `/docs/features/` - Feature Specifications
Contains numbered feature specifications (existing structure maintained):
- `0006_CROSS_LESSON_CONTEXT_SYSTEM.md` - Cross-lesson context system specification
- `0007_FLASHCARD_OPTIMIZATION_REFINEMENT.md` - Flashcard optimization feature
- `0009_QUIZLET_API_INTEGRATION.md` - Quizlet API integration specification

### `/docs/commands/` - Command Documentation
Contains documentation for specific commands and operations:
- `code_review.md` - Code review processes
- `create_brief.md` - Brief creation procedures
- `plan_feature.md` - Feature planning guidelines
- `write_docs.md` - Documentation writing standards

## File Naming Conventions

### Applied Naming Standards:
1. **Purpose-driven names**: Files clearly indicate their content and purpose
2. **Kebab-case formatting**: All lowercase with hyphens (e.g., `project-overview.md`)
3. **Descriptive prefixes**: Phase files include purpose description (e.g., `phase2a-drive-integration-implementation.md`)
4. **Logical grouping**: Related files are grouped in appropriate directories

### Previous vs. New Names:
- `PRODUCT_BRIEF.md` → `overview/project-overview.md`
- `DEVELOPMENT_PLAN.md` → `planning/development-plan.md`
- `PHASE1_IMPLEMENTATION_SUMMARY.md` → `phases/phase1-foundation-implementation.md`
- `FILE_COPYING_PERFORMANCE.md` → `implementation/file-copying-performance.md`
- `UI_SETUP_GUIDE.md` → `guides/ui-setup-guide.md`

## Benefits of New Organization

1. **Clear Purpose Identification**: Each directory name immediately indicates the type of content
2. **Logical Grouping**: Related documents are co-located for easy discovery
3. **Scalable Structure**: New documents can be easily categorized into existing directories
4. **AI-Friendly**: Other AI agents can quickly understand and navigate the documentation
5. **Maintenance-Friendly**: Updates and modifications are easier to locate and manage

## Usage Guidelines

- **For Project Overview**: Start with `/docs/overview/project-overview.md`
- **For Development Planning**: Check `/docs/planning/` directory
- **For Implementation Details**: Look in `/docs/phases/` for chronological development or `/docs/implementation/` for technical specifics
- **For Setup/Usage**: Consult `/docs/guides/` directory
- **For Feature Specs**: Browse `/docs/features/` for numbered specifications
- **For Command Reference**: Check `/docs/commands/` directory

This organization ensures that any AI agent or developer can quickly locate relevant documentation based on their specific needs and the type of information they're seeking.