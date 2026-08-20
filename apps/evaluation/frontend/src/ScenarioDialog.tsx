import './ScenarioDialog.css'

export type Scenario = {
  name: string
  role: string
  instructions: string
}

type Props = {
  customInstructions: string
  customRole: string
  defaultScenario: Scenario | null
  isLocked: boolean
  isOpen: boolean
  mode: 'default' | 'custom'
  onClose: () => void
  onCustomInstructionsChange: (value: string) => void
  onCustomRoleChange: (value: string) => void
  onModeChange: (mode: 'default' | 'custom') => void
}

function ScenarioDialog({
  customInstructions,
  customRole,
  defaultScenario,
  isLocked,
  isOpen,
  mode,
  onClose,
  onCustomInstructionsChange,
  onCustomRoleChange,
  onModeChange,
}: Props) {
  if (!isOpen || !defaultScenario) return null

  const scenario = mode === 'custom'
    ? { name: 'Custom scenario', role: customRole, instructions: customInstructions }
    : defaultScenario

  return (
    <div className="scenario-backdrop" role="presentation">
      <section className="scenario-dialog" aria-label="Evaluation scenario">
        <header>
          <div>
            <span>Controlled context</span>
            <h2>Evaluation scenario</h2>
          </div>
          <button aria-label="Close scenario" onClick={onClose} type="button">×</button>
        </header>

        <div className="scenario-mode-toggle" role="group" aria-label="Scenario source">
          <button className={mode === 'default' ? 'active' : ''} disabled={isLocked} onClick={() => onModeChange('default')} type="button">Default</button>
          <button className={mode === 'custom' ? 'active' : ''} disabled={isLocked} onClick={() => onModeChange('custom')} type="button">Custom</button>
        </div>

        <p className="scenario-note">
          {isLocked
            ? 'The same scenario is fixed across both active sessions.'
            : mode === 'custom'
              ? 'Custom text applies to the next Live A/B pair only. The shared default is unchanged.'
              : 'Both sides of a new Live A/B pair use this shared default. Replay always copies its source scenario.'}
        </p>

        <label className="scenario-field">
          <span>Role</span>
          <textarea
            maxLength={2000}
            onChange={(event) => onCustomRoleChange(event.target.value)}
            readOnly={isLocked || mode === 'default'}
            rows={4}
            value={scenario.role}
          />
        </label>
        <label className="scenario-field">
          <span>Instructions</span>
          <textarea
            maxLength={8000}
            onChange={(event) => onCustomInstructionsChange(event.target.value)}
            readOnly={isLocked || mode === 'default'}
            rows={10}
            value={scenario.instructions}
          />
        </label>

        <footer>
          {!isLocked && mode === 'custom' && (
            <button
              className="secondary"
              onClick={() => {
                onCustomRoleChange(defaultScenario.role)
                onCustomInstructionsChange(defaultScenario.instructions)
              }}
              type="button"
            >
              Reset custom text
            </button>
          )}
          <button onClick={onClose} type="button">Done</button>
        </footer>
      </section>
    </div>
  )
}

export default ScenarioDialog
