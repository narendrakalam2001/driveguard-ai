# DriveGuard AI — Traffic Violation FAQ (RAG Knowledge Base)

## What happens after a violation is detected?
The system captures an evidence frame, reads the number plate, classifies
severity (WARNING / VIOLATION / SEVERE) based on how far the speed exceeds
the posted limit, and generates a violation notice citing the relevant
Motor Vehicles Act section. The vehicle owner has 30 days to respond via
the nearest RTO before a formal challan is issued.

## How is severity determined?
- WARNING: speed exceeds the limit by more than 10%.
- VIOLATION: speed exceeds the limit by more than 20%.
- SEVERE: speed exceeds the limit by more than 50%.
Severity affects both the language of the notice and, in the real-world
deployment, whether the case is auto-escalated for manual police review.

## Does weather or road condition affect the notice?
Yes — heavy rain, fog, or a waterlogged road generally does not excuse a
speeding violation, but SEVERE violations recorded in poor weather are
noted as an aggravating factor in the generated notice, since Section 184
(dangerous driving) considers the "nature, condition and use of the place"
and prevailing conditions.

## What if the number plate cannot be read clearly?
The system flags the record as PLATE_UNCLEAR rather than guessing. Such
records are still logged with the evidence frame and speed data, but a
notice is only auto-generated once a human operator confirms the plate
(e.g. via the dashboard's evidence review screen) or a higher-confidence
re-read is obtained.

## Are repeat offenders treated differently?
The system checks the plate number against previously logged violations
at the same camera / citywide (deployment-dependent) and surfaces a
repeat-offender flag on the generated notice. Central Act penalties for
several sections step up materially for a second or subsequent offense —
see `fine_schedule.json` for the specific ranges used in this project's
generated notices.

## Is the fine amount shown on the notice legally binding?
No. The amounts in `fine_schedule.json` are indicative, drawn from the
central Motor Vehicles (Amendment) Act, 2019 schedule, for demonstration
purposes in this project. Real challans must reflect the current
state-notified amount, which can differ from the central schedule.
