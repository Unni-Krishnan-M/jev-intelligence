import { render } from "@testing-library/react";
import type { ReactElement } from "react";

import { TooltipProvider } from "@/components/ui/tooltip";

/** Render inside the providers the console's components expect (Radix tooltips). */
export function renderUi(ui: ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>);
}
