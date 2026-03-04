# Bulk Order Limits and Basket Risk Policy

## Scope
This policy defines quantity limits for bulk purchases, risk thresholds for perishable-heavy baskets, and the conditions under which orders are flagged for manual review before fulfillment.

## Standard Quantity Limits
The following per-SKU quantity caps apply to all standard consumer orders:

| Category | Max Units per Order |
|----------|-------------------|
| Shelf-stable grocery | 12 units |
| Beverages (non-alcoholic) | 6 units |
| Fresh produce | 6 units |
| Refrigerated items | 6 units |
| Frozen items | 4 units |
| Baby formula | 4 units |
| Cleaning and household | 8 units |

Orders exceeding these limits require manual review before fulfillment confirmation is sent to the customer.

## Perishable Bulk Risk
Orders where more than 60% of basket value consists of perishable items are classified as high-risk and subject to the following rules:

- Delivery must be assigned to Express or Standard windows only. Extended windows are blocked.
- The order is flagged in the fulfillment dashboard for prioritized pick scheduling.
- If any item in the perishable portion is substituted, the substitution must be reviewed by a senior picker before confirmation.

High-risk perishable baskets are defined as orders where the combined value of items from produce, dairy, meat, seafood, and deli departments exceeds $75 or 60% of total basket value, whichever threshold is reached first. Basket value refers to the pre-tax, pre-discount subtotal at the time of fulfillment pick.

## Basket Risk Flags
The following basket compositions trigger an automated risk flag in the order management system:

- **Temperature conflict**: Order contains both frozen and ambient items with no insulated packaging confirmed. If packaging status is unknown, flag the order for review rather than assuming compliance.
- **Volume-to-window mismatch**: Large order (20+ items) assigned to a 1-hour Express window.
- **Perishable concentration**: More than 8 perishable SKUs from high-risk categories in a single order.
- **Repeat bulk buyer**: Customer placing a bulk order for the same perishable SKU more than twice within 7 days.

Flagged orders are placed in a review queue. Fulfillment must not commence until the flag is cleared by a supervisor or the system auto-clears after validation.

## Business Account Exceptions
Verified business accounts (restaurants, catering, institutional buyers) are exempt from standard per-SKU caps subject to a signed bulk purchase agreement on file. These accounts are still subject to perishable delivery window restrictions and cold-chain handling requirements.
