-- platform/database/seed.sql

INSERT INTO projects (
    name,
    is_valid,
    quota,
    usage_count
)
VALUES (
    'Mini Project Template',
    TRUE,
    NULL,
    0
);

INSERT INTO access_keys (
    key_value,
    project_id,
    is_active,
    quota,
    usage_count
)
VALUES (
    'dev-template-key',
    (
        SELECT id
        FROM projects
        WHERE name = 'Mini Project Template'
    ),
    TRUE,
    NULL,
    0
);