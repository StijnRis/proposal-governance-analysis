# Database Schema Summary

## Table: Project
```sql
Rows: 1
Columns: 4

project_id INTEGER (PK)
project_name TEXT
enhancement_proposal_name TEXT
copyright TEXT
```

## Table: Person
```sql
Rows: 2119
Columns: 2

person_id INTEGER (PK)
full_name TEXT
```

## Table: sqlite_sequence
```sql
Rows: 4
Columns: 2

name 
seq 
```

## Table: Organisation
```sql
Rows: 0
Columns: 2

organisation_id INTEGER (PK)
organisation_name TEXT
```

## Table: PersonUsername
```sql
Rows: 707
Columns: 3

person_id INTEGER (PK)
domain TEXT (PK)
username TEXT (PK)
```

## Table: Affiliation
```sql
Rows: 0
Columns: 2

organisation_id INTEGER (PK)
person_id INTEGER (PK)
```

## Table: Proposal
```sql
Rows: 7691
Columns: 4

project_id INTEGER (PK)
proposal_id TEXT (PK)
topic TEXT
proposal_type TEXT
```

## Table: StageHistory
```sql
Rows: 5006
Columns: 6

project_id INTEGER (PK)
proposal_id TEXT (PK)
stage_index INTEGER (PK)
raw_status TEXT
normalised_status TEXT
created_at DATETIME
```

## Table: ProposalRevision
```sql
Rows: 11607
Columns: 7

project_id INTEGER (PK)
proposal_id TEXT (PK)
revision_index INTEGER (PK)
title TEXT
created_at DATETIME
content TEXT
implemented_at_version TEXT
```

## Table: ProposalRevisionAuthor
```sql
Rows: 18551
Columns: 4

project_id INTEGER (PK)
proposal_id TEXT (PK)
revision_index INTEGER (PK)
author_id INTEGER (PK)
```

## Table: RelatedProposal
```sql
Rows: 0
Columns: 5

project_id INTEGER (PK)
proposal_id TEXT (PK)
related_project_id INTEGER (PK)
related_proposal_id TEXT (PK)
type TEXT
```

## Table: Comment
```sql
Rows: 18143
Columns: 7

comment_id INTEGER (PK)
author_id INTEGER
project_id INTEGER
proposal_id TEXT
comment_on_comment_id INTEGER
created_at DATETIME
content TEXT
```

