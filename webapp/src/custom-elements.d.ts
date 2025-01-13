declare global {
    namespace JSX {
      interface IntrinsicElements {
        "skyview-app": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
          embed?: string;
        };
      }
    }
  }
  
  export {};
  