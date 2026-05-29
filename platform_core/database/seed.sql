-- platform/database/seed.sql

INSERT INTO projects (
    name,
    is_valid,
    quota,
    usage_count
)
SELECT
    'Mini Project Template',
    TRUE,
    NULL,
    0
WHERE NOT EXISTS (
    SELECT 1
    FROM projects
    WHERE name = 'Mini Project Template'
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
        ORDER BY created_at ASC
        LIMIT 1
    ),
    TRUE,
    NULL,
    0
)
ON CONFLICT (key_value) DO NOTHING;
