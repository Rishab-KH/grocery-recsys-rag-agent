# Snacks and Shelf-Stable Department Handling Policy

## Scope
This policy covers quality handling, promotional sensitivity, and bulk purchase tolerance for the snacks and shelf-stable grocery department. This department has the highest substitution tolerance and the lowest spoilage risk of any category in the fulfillment workflow.

## Definitions

- **Sell-by date**: The date after which the retailer should no longer display the product for sale. Used as the operative shelf-life cutoff in all fulfillment decisions in this policy.
- **Best-by date**: A manufacturer's quality estimate, not a safety limit. Best-by dates are informational and do not independently disqualify an item from fulfillment, provided the sell-by date has not passed.

## Shelf-Stable Handling Standards
Snacks and shelf-stable items do not require temperature-controlled staging or transport. Standard ambient handling applies with the following minimum requirements:

- Items must be inspected for damaged packaging — crushed boxes, punctured bags, and broken seals disqualify an item from fulfillment.
- Items must be within sell-by date with a minimum of 14 days' remaining shelf life at the time of pick.
- Canned goods must not show dents at the seam or lid. Body dents (non-seam) are acceptable if the can is otherwise intact.
- Bulk multipacks must be fulfilled as complete units. Partial multipacks may not be substituted for a full multipack.

## Substitution Tolerance
Snacks and shelf-stable items have high substitution tolerance. The following substitutions are permitted:

- Same flavor/variety, different brand — permitted for commodity snack categories (chips, crackers, cookies, nuts).
- Same brand, adjacent variety — permitted when the specific flavor or variety is unavailable (e.g., original flavor substituted with lightly salted).
- Different format, same category — permitted only when the size is equivalent or larger (e.g., single-serve bag may not substitute for a family-size bag; family-size may substitute for single-serve only at adjusted pricing; price adjustment is handled by the pricing engine, not by the picker).

Substitutions that change the allergen profile of the item are not permitted. If a customer's order contains an allergen-flagged item, substitution must be limited to items with an identical allergen declaration. If the allergen declaration of a candidate substitute is missing or unclear, do not auto-substitute; leave the slot empty and notify the customer.

## Promotional Sensitivity
Snacks is one of the highest-volume promotional categories. The following rules apply:

- Promotional snack items must be picked as specified. Substituting a promotional SKU with a non-promotional SKU requires the promotional price to be honored on the substitute only if the promotion is configured as transferable in the promotions system.
- During high-volume promotional events (major holidays, sporting events), snack SKUs on promotion are subject to the bulk limits defined in `bulk_limits.md`. Maximum 6 units per promotional SKU per order.
- Promotional multipacks may not be broken apart for fulfillment of single-unit orders.

## Bulk Tolerance
Snacks and shelf-stable items have the highest bulk tolerance of any department:

- Standard per-SKU limit is 12 units. This may be raised to 24 units for verified business accounts.
- Bulk snack orders do not require the high-risk basket review triggered for perishable-heavy baskets.
- Large snack orders (15+ SKUs from the snacks aisle) should be consolidated in fulfillment sequencing to reduce pick time, but this carries no additional compliance requirement.
