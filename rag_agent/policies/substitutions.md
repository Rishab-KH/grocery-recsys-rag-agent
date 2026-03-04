# Product Substitution Policy

## Scope
This policy governs all automated and agent-assisted product substitutions triggered when a requested item is unavailable at the time of order fulfillment. It applies to all order channels and fulfillment centers.

## When Substitutions Are Permitted
Substitutions may be initiated under the following conditions:
- The requested item is marked out-of-stock, or low-stock where fulfillment risk exceeds threshold at the time of pick.
- The customer has not explicitly opted out of substitutions at the account or order level.
- The substitution candidate meets all active quality, safety, and compliance standards.

Substitutions are not permitted when:
- The customer has flagged the item as allergy-sensitive or dietary-restricted.
- The item is part of a promotional bundle requiring the specific SKU.
- No eligible substitute exists within the approved aisle or department.

## Department and Aisle Preference Rules
Substitutes must be selected in the following priority order:

1. **Same aisle, same brand** — preferred when size variance is within 10% of original.
2. **Same aisle, different brand** — permitted when brand is not flagged as customer-preferred.
3. **Same department, adjacent aisle** — permitted only for shelf-stable categories.
4. **Cross-department** — not permitted under any automated workflow; requires manual override and supervisor approval.

Substitutions that cross department boundaries are logged as exceptions and flagged for post-order audit.

## Brand and Size Sensitivity
Brand substitutions require additional caution in the following categories:
- Baby products and infant formula — no substitution without explicit customer consent.
- Organic-labeled items — may only be substituted with other certified organic products.
- Private label items — may be substituted with equivalent national brand at no additional cost to the customer.

Size substitutions that result in a smaller unit quantity are prohibited. Upsizing (larger volume at same or adjusted price) is permitted per the pricing adjustment rules defined in `promo_rules.md`.

## Customer Communication Requirements
When a substitution is applied, the following communications are mandatory:

- **Pre-delivery notification**: Customer must receive itemized substitution details via SMS or push notification no later than 30 minutes before the scheduled delivery window.
- **Receipt disclosure**: The final receipt must clearly distinguish substituted items from originally ordered items, including price differential if applicable.
- **Opt-out enforcement**: If a customer rejects a substitution before delivery dispatch, the item must be removed from the order and the customer notified of the resulting gap.

All substitution events must be logged in the order management system with: original SKU, substitute SKU, reason code, timestamp, and fulfillment agent ID.
