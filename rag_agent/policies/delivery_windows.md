# Delivery Window and Perishable Timing Policy

## Scope
This policy defines standards for delivery scheduling with specific reference to perishable risk, spoilage thresholds, and peak-window capacity management.

## Definitions

- **Window assignment time**: The time at which a delivery window is confirmed to the customer, typically at order placement.
- **Window start time**: The scheduled time at which the delivery driver departs the fulfillment center for the first stop on the route.
- **Delivery window breach**: A delay that causes the actual delivery to exceed the customer's assigned window end time by any amount.

## Delivery Window Definitions
Orders are classified into one of three window tiers:

| Tier | Window Duration | Perishable Eligibility |
|------|----------------|------------------------|
| Express | 1–2 hours | Fully eligible |
| Standard | 2–4 hours | Eligible with packing guidelines |
| Extended | 4+ hours | Restricted for high-risk perishables |

Orders containing refrigerated or frozen items must not be assigned to Extended windows unless the customer has been informed and has explicitly accepted the spoilage risk disclosure. If the customer's disclosure acceptance status cannot be confirmed, the order must not be assigned to an Extended window.

## Window Eligibility by Perishable Risk

| Category Risk | Express (1–2h) | Standard (2–4h) | Extended (4+h) |
|---|---|---|---|
| High risk (raw meat, seafood, cut fruit, fresh juice) | Eligible | Eligible with ice pack | Prohibited |
| Moderate risk (whole produce, deli, prepared meals) | Eligible | Eligible | Restricted — customer disclosure required |
| Low risk (hard cheese, whole eggs, shelf-stable dairy) | Eligible | Eligible | Eligible |
| Frozen | Eligible | Eligible with active vehicle refrigeration | Prohibited |

## Spoilage Risk by Category
The following categories carry elevated spoilage risk and must be flagged during window assignment:

- **High risk**: Raw meat, seafood, cut fruit, fresh juices, unpasteurized dairy.
- **Moderate risk**: Whole produce, deli items, prepared meals, fresh pasta.
- **Low risk**: Hard cheeses, whole eggs, shelf-stable dairy alternatives.

High-risk items should not be included in orders assigned to Standard windows during summer months (June–August) unless insulated packaging is confirmed available at the fulfillment center. If packaging availability is unknown, treat as unavailable and restrict the order accordingly.

## Peak Window Considerations
Peak delivery windows (Friday 4–8 PM, Saturday 10 AM–2 PM, Sunday 9 AM–1 PM) are subject to the following additional constraints:

- Order batching for perishables is limited to a maximum of three orders per route during peak windows.
- Substitution eligibility for perishable items is reduced: only same-aisle substitutes are permitted during peak periods to minimize picker time.
- Drivers must complete perishable deliveries before non-perishable stops on the same route.

## Perishable Packing Guidance
- Refrigerated items must be packed in insulated bags with a minimum of one ice pack per bag, regardless of ambient temperature.
- Raw proteins must be bagged separately from produce and ready-to-eat items.
- Frozen items must be co-located with dry ice packs where available; standard ice packs are not sufficient for frozen transport exceeding 90 minutes. If dry ice is unavailable, frozen orders with a transit time exceeding 90 minutes must be held until supply is restored or rerouted to a shorter delivery sequence.

## Escalation
Orders where the delivery window is extended post-dispatch due to route delay must trigger an automated customer notification. If the revised delivery time exceeds the original window by more than the SLA threshold (e.g., 45 minutes), a spoilage credit equal to the value of affected perishables must be pre-authorized in the order management system.
