# Promotional Pricing and Eligibility Policy

## Scope
This policy governs promotional pricing eligibility, how promotions interact with substitution workflows, and the disclosure requirements when promotional items are unavailable or substituted.

## Promotional Pricing Eligibility
A promotional price applies to a specific SKU during a defined promotional window. The following conditions must be met for a promotional price to be honored:

- The item is in the customer's cart at the time the order is placed within the promotional window.
- The promotional SKU is in stock at the time of fulfillment pick.
- The promotion has not been flagged as suspended due to supplier or pricing system error.

Promotions do not carry forward to substituted items unless explicitly configured in the promotion management system as transferable.

## Substitution Pricing Implications
When a promotional item is substituted due to stockout, the following pricing rules apply:

- **Non-transferable promotion**: The customer is charged the standard price of the substitute item. The promotional item is removed from the receipt and the customer is notified.
- **Transferable promotion**: The promotional discount is applied to the substitute item, provided the substitute is in the same category and its standard price does not exceed 120% of the original promotional price.
- **Customer-preferred brand on promotion**: If the promoted item is marked as a customer-preferred brand in their account profile, no automatic substitution is applied. The slot is left empty and the customer notified.

Price adjustments resulting from substitution are applied at order close, not at pick time, to ensure accurate billing.

## Promotional Outcome Decision Table

| Scenario | Promo Applied? | Substitute Used? | Customer Notified? |
|---|---|---|---|
| Promotional SKU in stock at pick | Yes — full promotional price | No substitution needed | No |
| OOS — promotion non-transferable | No — substitute billed at standard price | Yes, any eligible substitute | Yes — before dispatch |
| OOS — promotion transferable (substitute ≤ 120% original price) | Yes — discount transferred to substitute | Yes, same-category substitute | Yes — receipt shows transfer |
| Bundle partially fulfilled (not all units available) | No — full bundle price invalidated; standard price applies to available unit | No bundle substitution | Yes — bundle outcome disclosed |
| OOS — customer-preferred brand on promotion | No | No — slot left empty | Yes — before dispatch |

## Bundle and Multi-Buy Promotions
For promotions requiring the purchase of multiple units (e.g., "Buy 2 get 1 free"):

- All units in the bundle must be available for the promotional price to apply.
- Partial fulfillment of a bundle (e.g., only 1 of 2 required units available) invalidates the promotion. The customer is charged standard price for the available unit.
- Substituting one unit of a bundle with a different SKU invalidates the multi-buy structure unless the substitute is explicitly listed as an eligible bundle participant in the promotion configuration.

## Customer Disclosure Requirements
When a promotion cannot be honored due to stockout or ineligible substitution:

- The customer must receive notification before delivery dispatch listing the original promotional item, its promotional price, and the outcome (substitution at standard price, or item dropped).
- The final receipt must clearly show the original promotional SKU as unavailable and the applied substitute with its pricing.
- Automated promotional discount reversals must be reflected in the customer's final charge within the SLA window (e.g., 15 minutes) of order close.

## Audit and Compliance
All promotional pricing exceptions — including manual overrides, transferable promotions applied, and bundle invalidations — must be logged in the promotions audit trail with SKU, order ID, applied rule, and agent ID. This log is reviewed weekly by the pricing compliance team.
