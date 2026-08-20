import './ScenarioDialog.css'

type Scenario = {
  name: string
  role: string
  instructions: string
}

type Props = {
  activeScenario: Scenario | null
  customInstructions: string
  customRole: string
  defaultScenario: Scenario | null
  isActiveSession: boolean
  isOpen: boolean
  mode: 'default' | 'custom'
  onClose: () => void
  onCustomInstructionsChange: (value: string) => void
  onCustomRoleChange: (value: string) => void
  onModeChange: (mode: 'default' | 'custom') => void
}

function ScenarioDialog({
  activeScenario,
  customInstructions,
  customRole,
  defaultScenario,
  isActiveSession,
  isOpen,
  mode,
  onClose,
  onCustomInstructionsChange,
  onCustomRoleChange,
  onModeChange,
}: Props) {
  const scenario = isActiveSession && activeScenario
    ? activeScenario
    : mode === 'custom' && defaultScenario
      ? { name: 'Custom scenario', role: customRole, instructions: customInstructions }
      : defaultScenario

  if (!isOpen || !scenario) return null

  return (
    <div className="scenario-backdrop" role="presentation">
      <section className="scenario-dialog" aria-label="Chat scenario">
        <header>
          <div>
            <span>Session context</span>
            <h2>{isActiveSession ? 'Active session scenario' : 'Scenario for the next chat'}</h2>
          </div>
          <button aria-label="Close scenario" onClick={onClose} type="button">×</button>
        </header>

        {!isActiveSession && (
          <div className="scenario-mode-toggle" role="group" aria-label="Scenario source">
            <button className={mode === 'default' ? 'active' : ''} onClick={() => onModeChange('default')} type="button">Default</button>
            <button className={mode === 'custom' ? 'active' : ''} onClick={() => onModeChange('custom')} type="button">Custom</button>
          </div>
        )}

        <p className="scenario-note">
          {isActiveSession
            ? 'This scenario is fixed for the active session.'
            : mode === 'custom'
              ? 'Custom text applies only to the next session. The system default is unchanged.'
              : 'This is the shared system default for new sessions.'}
        </p>

        <label className="scenario-field">
          <span>Role</span>
          <textarea
            maxLength={2000}
            onChange={(event) => onCustomRoleChange(event.target.value)}
            readOnly={isActiveSession || mode === 'default'}
            rows={4}
            value={scenario.role}
          />
        </label>
        <label className="scenario-field">
          <span>Instructions</span>
          <textarea
            maxLength={8000}
            onChange={(event) => onCustomInstructionsChange(event.target.value)}
            readOnly={isActiveSession || mode === 'default'}
            rows={10}
            value={scenario.instructions}
          />
        </label>

        <footer>
          {!isActiveSession && mode === 'custom' && defaultScenario && (
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
